import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import json
import logging
from datetime import datetime, timedelta
import sys
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("uk_covid_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("uk_covid_viz")

class UKCovidDataVisualizer:
    """A class to fetch, process, and visualize UK COVID-19 and vaccine data."""
    
    def __init__(self, data_dir="data"):
        """Initialize the class with data directory and API URL."""
        self.data_dir = data_dir
        self.uk_api_url = "https://api.coronavirus.data.gov.uk/v1/data"
        self.uk_dashboard_url = "https://coronavirus.data.gov.uk/details/download"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Set the plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Set the seed for reproducible sample data
        np.random.seed(42)
        
    def fetch_covid_data(self, area_type="overview", area_name=None, metric_list=None, days=90):
        """
        Fetch COVID-19 data from the UK government API.
        
        Args:
            area_type (str): Type of area to filter data by (e.g., 'overview', 'nation', 'region')
            area_name (str): Name of the area if not using 'overview'
            metric_list (list): List of metrics to retrieve
            days (int): Number of days of data to fetch
            
        Returns:
            pandas.DataFrame: DataFrame containing the requested COVID-19 data
        """
        if metric_list is None:
            metric_list = [
                "newCasesByPublishDate",
                "cumCasesByPublishDate",
                "newDeaths28DaysByPublishDate",
                "cumDeaths28DaysByPublishDate",
                "newPeopleVaccinatedFirstDoseByPublishDate",
                "cumPeopleVaccinatedFirstDoseByPublishDate",
                "newPeopleVaccinatedSecondDoseByPublishDate",
                "cumPeopleVaccinatedSecondDoseByPublishDate",
                "newPeopleVaccinatedThirdInjectionByPublishDate",
                "cumPeopleVaccinatedThirdInjectionByPublishDate"
            ]
            
        try:
            # Construct the filters parameter
            filters = [f"areaType={area_type}"]
            if area_name:
                filters.append(f"areaName={area_name}")
                
            # Format the API request
            api_params = {
                "filters": ";".join(filters),
                "structure": json.dumps({
                    "date": "date",
                    "areaName": "areaName",
                    "areaCode": "areaCode",
                    **{metric: metric for metric in metric_list}
                }),
                "latestBy": None,
                "format": "json"
            }
            
            logger.info(f"Fetching data from UK government API for the last {days} days")
            response = requests.get(self.uk_api_url, params=api_params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["data"])
                
                # Convert date column to datetime
                df["date"] = pd.to_datetime(df["date"])
                
                # Sort by date
                df = df.sort_values("date").reset_index(drop=True)
                
                # Keep only the specified number of days
                if days and len(df) > days:
                    df = df.tail(days)
                    
                # Some metrics might be missing, replace NaN with 0
                df = df.fillna(0)
                
                logger.info(f"Successfully fetched {len(df)} days of data")
                return df
                
            elif response.status_code == 404:
                logger.error(f"API endpoint not found (404). The UK COVID API may have been discontinued or relocated.")
                return self._generate_sample_data(days)
            elif response.status_code == 403:
                logger.error(f"Access forbidden (403). API access may require authorization now.")
                return self._generate_sample_data(days)
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                raise Exception(f"API request failed with status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error occurred: {str(e)}")
            logger.warning("API unavailable - falling back to sample data generation")
            return self._generate_sample_data(days)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            logger.warning("API data parsing failed - falling back to sample data generation")
            return self._generate_sample_data(days)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            logger.warning("Unexpected error - falling back to sample data generation")
            return self._generate_sample_data(days)
            
    def save_data(self, df, filename):
        """Save the DataFrame to a CSV file."""
        try:
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise
            
    def load_data(self, filename):
        """Load data from a CSV file."""
        try:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"File {filepath} does not exist")
                return None
                
            df = pd.read_csv(filepath)
            
            # Convert date column back to datetime if it exists
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                
            logger.info(f"Data loaded from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
            
    def plot_case_data(self, df, output_dir="plots"):
        """Plot COVID-19 case data."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot new cases
            if "newCasesByPublishDate" in df.columns:
                ax.bar(df["date"], df["newCasesByPublishDate"], alpha=0.6, color="steelblue", label="New Cases")
                
                # Add 7-day rolling average
                df["rolling_avg_cases"] = df["newCasesByPublishDate"].rolling(window=7).mean()
                ax.plot(df["date"], df["rolling_avg_cases"], color="darkblue", linewidth=2, label="7-Day Average")
                
                # Format the plot
                ax.set_title("UK Daily COVID-19 Cases", fontsize=16)
                ax.set_xlabel("Date", fontsize=14)
                ax.set_ylabel("Number of Cases", fontsize=14)
                
                # Format x-axis date labels
                date_form = DateFormatter("%Y-%m-%d")
                ax.xaxis.set_major_formatter(date_form)
                plt.xticks(rotation=45)
                
                # Add grid lines
                ax.grid(True, linestyle="--", alpha=0.7)
                
                # Add legend
                ax.legend()
                
                # Tight layout
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(os.path.join(output_dir, "uk_covid_cases.png"), dpi=300)
                plt.close()
                
                logger.info("Case data plot created and saved")
            else:
                logger.warning("New cases data not available for plotting")
                
        except Exception as e:
            logger.error(f"Error plotting case data: {str(e)}")
            raise
            
    def plot_vaccine_data(self, df, output_dir="plots"):
        """Plot COVID-19 vaccine data."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if vaccine data is available
            vaccine_columns = [
                "cumPeopleVaccinatedFirstDoseByPublishDate",
                "cumPeopleVaccinatedSecondDoseByPublishDate",
                "cumPeopleVaccinatedThirdInjectionByPublishDate"
            ]
            
            if not any(col in df.columns for col in vaccine_columns):
                logger.warning("Vaccine data not available for plotting")
                return
                
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot cumulative vaccinations
            for col, (label, color) in zip(
                vaccine_columns,
                [("First Dose", "green"), ("Second Dose", "blue"), ("Booster", "purple")]
            ):
                if col in df.columns:
                    ax.plot(df["date"], df[col], color=color, linewidth=2, label=label)
                    
            # Format the plot
            ax.set_title("UK COVID-19 Vaccination Progress", fontsize=16)
            ax.set_xlabel("Date", fontsize=14)
            ax.set_ylabel("Cumulative Vaccinations", fontsize=14)
            
            # Format y-axis as millions
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
            
            # Format x-axis date labels
            date_form = DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            plt.xticks(rotation=45)
            
            # Add grid lines
            ax.grid(True, linestyle="--", alpha=0.7)
            
            # Add legend
            ax.legend()
            
            # Tight layout
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, "uk_vaccination_progress.png"), dpi=300)
            plt.close()
            
            logger.info("Vaccine data plot created and saved")
            
        except Exception as e:
            logger.error(f"Error plotting vaccine data: {str(e)}")
            raise
            
    def plot_combined_metrics(self, df, output_dir="plots"):
        """Plot cases vs. deaths for comparison."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if required columns are available
            if "newCasesByPublishDate" not in df.columns or "newDeaths28DaysByPublishDate" not in df.columns:
                logger.warning("Cases or deaths data not available for comparison plotting")
                return
                
            # Create a figure with two y-axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot cases on the first y-axis
            ax1.bar(df["date"], df["newCasesByPublishDate"], alpha=0.4, color="steelblue", label="New Cases")
            ax1.set_ylabel("New Cases", fontsize=14, color="steelblue")
            ax1.tick_params(axis="y", labelcolor="steelblue")
            
            # Plot deaths on the second y-axis
            ax2.bar(df["date"], df["newDeaths28DaysByPublishDate"], alpha=0.6, color="crimson", label="New Deaths")
            ax2.set_ylabel("New Deaths", fontsize=14, color="crimson")
            ax2.tick_params(axis="y", labelcolor="crimson")
            
            # Add 7-day rolling averages
            df["rolling_avg_cases"] = df["newCasesByPublishDate"].rolling(window=7).mean()
            df["rolling_avg_deaths"] = df["newDeaths28DaysByPublishDate"].rolling(window=7).mean()
            
            ax1.plot(df["date"], df["rolling_avg_cases"], color="darkblue", linewidth=2, label="7-Day Avg Cases")
            ax2.plot(df["date"], df["rolling_avg_deaths"], color="darkred", linewidth=2, label="7-Day Avg Deaths")
            
            # Format the plot
            ax1.set_title("UK COVID-19 Cases vs. Deaths", fontsize=16)
            ax1.set_xlabel("Date", fontsize=14)
            
            # Format x-axis date labels
            date_form = DateFormatter("%Y-%m-%d")
            ax1.xaxis.set_major_formatter(date_form)
            plt.xticks(rotation=45)
            
            # Add grid lines
            ax1.grid(True, linestyle="--", alpha=0.7)
            
            # Add legends for both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            
            # Tight layout
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, "uk_cases_vs_deaths.png"), dpi=300)
            plt.close()
            
            logger.info("Combined metrics plot created and saved")
            
        except Exception as e:
            logger.error(f"Error plotting combined metrics: {str(e)}")
            raise
            
    def _generate_sample_data(self, days=90):
        """Generate sample COVID-19 and vaccine data when the API is unavailable.
        
        Args:
            days (int): Number of days of data to generate
            
        Returns:
            pandas.DataFrame: DataFrame containing sample COVID-19 data
        """
        logger.info(f"Generating sample data for {days} days")
        
        # Generate date range ending today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days-1)
        dates = pd.date_range(start=start_date, end=end_date, periods=days)
        
        # Create base dataframe with dates
        df = pd.DataFrame({'date': dates})
        
        # Add area information
        df['areaName'] = 'United Kingdom'
        df['areaCode'] = 'UK'
        
        # Generate synthetic case data with a declining trend and weekly seasonality
        base_cases = np.linspace(20000, 5000, days)  # Declining trend
        weekly_pattern = 0.3 * np.sin(np.linspace(0, days/7 * 2 * np.pi, days))  # Weekly pattern
        noise = np.random.normal(0, 0.1, days)  # Random noise
        
        # Combine components and ensure positive values
        case_series = base_cases * (1 + weekly_pattern + noise)
        df['newCasesByPublishDate'] = np.maximum(100, case_series.astype(int))
        
        # Calculate cumulative cases
        df['cumCasesByPublishDate'] = df['newCasesByPublishDate'].cumsum() + 17000000  # Starting from a high base
        
        # Generate death data (correlated with cases but with lag and lower numbers)
        deaths_base = base_cases * 0.005  # Death rate declining over time due to vaccines
        deaths_lagged = np.roll(deaths_base, 14)  # 14-day lag from cases to deaths
        deaths_lagged[:14] = deaths_lagged[14] * np.linspace(1.5, 1.0, 14)  # Fill in the first 14 days
        deaths_noise = np.random.normal(0, 0.1, days)
        df['newDeaths28DaysByPublishDate'] = np.maximum(0, (deaths_lagged * (1 + deaths_noise)).astype(int))
        
        # Calculate cumulative deaths
        df['cumDeaths28DaysByPublishDate'] = df['newDeaths28DaysByPublishDate'].cumsum() + 175000  # Starting from a high base
        
        # Generate vaccine data - first doses (slowing down as approaching saturation)
        first_dose_daily_base = np.linspace(100000, 10000, days)  # Declining as more people are vaccinated
        first_dose_noise = np.random.normal(0, 0.2, days)
        df['newPeopleVaccinatedFirstDoseByPublishDate'] = np.maximum(5000, (first_dose_daily_base * (1 + first_dose_noise)).astype(int))
        
        # First dose - cumulative (approaching saturation around 90% of population)
        uk_population = 67000000
        first_dose_target = 0.9 * uk_population  # 90% of population
        first_dose_base = 0.85 * uk_population  # Starting at 85% already vaccinated
        df['cumPeopleVaccinatedFirstDoseByPublishDate'] = first_dose_base + df['newPeopleVaccinatedFirstDoseByPublishDate'].cumsum()
        
        # Cap at target
        df['cumPeopleVaccinatedFirstDoseByPublishDate'] = np.minimum(first_dose_target, df['cumPeopleVaccinatedFirstDoseByPublishDate'])
        
        # Second doses (similar pattern but slightly lower)
        second_dose_daily_base = np.linspace(80000, 8000, days)
        second_dose_noise = np.random.normal(0, 0.2, days)
        df['newPeopleVaccinatedSecondDoseByPublishDate'] = np.maximum(4000, (second_dose_daily_base * (1 + second_dose_noise)).astype(int))
        
        # Second dose - cumulative (following first dose pattern with a delay)
        second_dose_target = 0.87 * uk_population  # 87% of population
        second_dose_base = 0.82 * uk_population  # Starting at 82% already vaccinated
        df['cumPeopleVaccinatedSecondDoseByPublishDate'] = second_dose_base + df['newPeopleVaccinatedSecondDoseByPublishDate'].cumsum()
        
        # Cap at target
        df['cumPeopleVaccinatedSecondDoseByPublishDate'] = np.minimum(second_dose_target, df['cumPeopleVaccinatedSecondDoseByPublishDate'])
        
        # Booster doses
        booster_daily_base = np.linspace(60000, 15000, days)
        booster_noise = np.random.normal(0, 0.3, days)
        df['newPeopleVaccinatedThirdInjectionByPublishDate'] = np.maximum(7000, (booster_daily_base * (1 + booster_noise)).astype(int))
        
        # Booster - cumulative
        booster_target = 0.7 * uk_population  # 70% of population
        booster_base = 0.6 * uk_population  # Starting at 60% already vaccinated
        df['cumPeopleVaccinatedThirdInjectionByPublishDate'] = booster_base + df['newPeopleVaccinatedThirdInjectionByPublishDate'].cumsum()
        
        # Cap at target
        df['cumPeopleVaccinatedThirdInjectionByPublishDate'] = np.minimum(booster_target, df['cumPeopleVaccinatedThirdInjectionByPublishDate'])
        
        logger.info("Sample data generation completed")
        
        return df
        
    def plot_vaccine_impact(self, df, output_dir="plots"):
        """Plot the impact of vaccination on case fatality rate."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if required columns are available
            required_cols = [
                "newCasesByPublishDate", 
                "newDeaths28DaysByPublishDate",
                "cumPeopleVaccinatedFirstDoseByPublishDate"
            ]
            
            if not all(col in df.columns for col in required_cols):
                logger.warning("Required data for vaccine impact analysis not available")
                return
                
            # Calculate case fatality rate (CFR) - 14-day lag for deaths relative to cases
            # Create a copy to avoid warnings about setting values on a slice
            df_cfr = df.copy()
            
            # Compute 14-day rolling sum of cases and deaths to smooth out reporting fluctuations
            df_cfr["cases_14d_sum"] = df_cfr["newCasesByPublishDate"].rolling(window=14).sum()
            df_cfr["deaths_14d_sum"] = df_cfr["newDeaths28DaysByPublishDate"].rolling(window=14).sum()
            
            # Shift cases back by 14 days to account for lag between infection and potential death
            df_cfr["cases_14d_sum_lagged"] = df_cfr["cases_14d_sum"].shift(-14)
            
            # Calculate CFR (deaths / lagged cases)
            df_cfr["cfr"] = (df_cfr["deaths_14d_sum"] / df_cfr["cases_14d_sum_lagged"]) * 100
            
            # Calculate vaccination rate (% of population with at least first dose)
            # Assuming UK population of approximately 67 million
            uk_population = 67000000
            df_cfr["vaccination_rate"] = (df_cfr["cumPeopleVaccinatedFirstDoseByPublishDate"] / uk_population) * 100
            
            # Remove rows with NaN values after calculations
            df_cfr = df_cfr.dropna(subset=["cfr", "vaccination_rate"])
            
            if len(df_cfr) < 10:
                logger.warning("Insufficient data points for vaccine impact analysis after preprocessing")
                return
                
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot CFR on the first y-axis
            cfr_color = "crimson"
            ax1.plot(df_cfr["date"], df_cfr["cfr"], color=cfr_color, linewidth=2, label="Case Fatality Rate (%)")
            ax1.set_ylabel("Case Fatality Rate (%)", fontsize=14, color=cfr_color)
            ax1.tick_params(axis="y", labelcolor=cfr_color)
            
            # Plot vaccination rate on the second y-axis
            vax_color = "forestgreen"
            ax2.plot(df_cfr["date"], df_cfr["vaccination_rate"], color=vax_color, linewidth=2, 
                     label="Population with First Dose (%)")
            ax2.set_ylabel("Vaccination Rate (%)", fontsize=14, color=vax_color)
            ax2.tick_params(axis="y", labelcolor=vax_color)
            
            # Format the plot
            ax1.set_title("UK COVID-19 Case Fatality Rate vs. Vaccination Progress", fontsize=16)
            ax1.set_xlabel("Date", fontsize=14)
            
            # Format x-axis date labels
            date_form = DateFormatter("%Y-%m-%d")
            ax1.xaxis.set_major_formatter(date_form)
            plt.xticks(rotation=45)
            
            # Add grid lines
            ax1.grid(True, linestyle="--", alpha=0.7)
            
            # Add legends for both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            
            # Add explanatory annotation
            plt.figtext(0.5, 0.01, 
                       "Note: CFR calculated using 14-day rolling sums with a 14-day lag between cases and deaths", 
                       ha="center", fontsize=10, style="italic")
            
            # Tight layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, "uk_vaccine_impact.png"), dpi=300)
            plt.close()
            
            logger.info("Vaccine impact plot created and saved")
            
        except Exception as e:
            logger.error(f"Error plotting vaccine impact: {str(e)}")
            raise
            
    def run_visualization_pipeline(self, days=90, use_cached=True):
        """Run the complete data fetching and visualization pipeline."""
        try:
            data_filename = f"uk_covid_data_{days}days.csv"
            
            # Try to load cached data if available and requested
            df = None
            if use_cached:
                df = self.load_data(data_filename)
                
            # Fetch fresh data if no cached data or it's too old
            if df is None or (
                df is not None and 
                datetime.now() - df["date"].max() > timedelta(days=1)
            ):
                logger.info("Fetching fresh data from API")
                df = self.fetch_covid_data(days=days)
                self.save_data(df, data_filename)
            else:
                logger.info("Using cached data")
                
            # Generate all plots
            self.plot_case_data(df)
            self.plot_vaccine_data(df)
            self.plot_combined_metrics(df)
            self.plot_vaccine_impact(df)
            
            logger.info("Visualization pipeline completed successfully")
            
            return {
                "data_file": os.path.join(self.data_dir, data_filename),
                "plot_files": [
                    os.path.join("plots", "uk_covid_cases.png"),
                    os.path.join("plots", "uk_vaccination_progress.png"),
                    os.path.join("plots", "uk_cases_vs_deaths.png"),
                    os.path.join("plots", "uk_vaccine_impact.png")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in visualization pipeline: {str(e)}")
            raise


def main():
    """Main function to run the visualization tool."""
    try:
        print("UK COVID-19 and Vaccine Data Visualization Tool")
        print("==============================================")
        
        # Initialize the visualizer
        visualizer = UKCovidDataVisualizer()
        
        # Parse command-line arguments if provided
        days = 90  # Default to 90 days
        use_cached = True
        use_sample_data = False
        
        if len(sys.argv) > 1:
            if sys.argv[1].lower() == "sample":
                use_sample_data = True
                print("Using sample data mode (will not attempt to contact the API)")
            else:
                try:
                    days = int(sys.argv[1])
                except ValueError:
                    print(f"Invalid number of days: {sys.argv[1]}. Using default (90 days).")
                
        if len(sys.argv) > 2:
            use_cached = sys.argv[2].lower() not in ("false", "no", "0", "f", "n")
            
        print(f"Running visualization for the last {days} days (Use cached: {use_cached})")
        
        # Run the visualization pipeline
        if use_sample_data:
            # Generate sample data directly
            sample_data = visualizer._generate_sample_data(days)
            sample_filename = f"uk_covid_sample_data_{days}days.csv"
            visualizer.save_data(sample_data, sample_filename)
            
            # Generate visualizations from sample data
            visualizer.plot_case_data(sample_data)
            visualizer.plot_vaccine_data(sample_data)
            visualizer.plot_combined_metrics(sample_data)
            visualizer.plot_vaccine_impact(sample_data)
            
            result = {
                "data_file": os.path.join(visualizer.data_dir, sample_filename),
                "plot_files": [
                    os.path.join("plots", "uk_covid_cases.png"),
                    os.path.join("plots", "uk_vaccination_progress.png"),
                    os.path.join("plots", "uk_cases_vs_deaths.png"),
                    os.path.join("plots", "uk_vaccine_impact.png")
                ]
            }
            print("\nNote: Using generated sample data instead of real API data")
        else:
            # Run the regular pipeline with API fallback
            result = visualizer.run_visualization_pipeline(days=days, use_cached=use_cached)
        
        print("\nVisualization completed successfully!")
        print(f"Data saved to: {result['data_file']}")
        print("Plots saved to:")
        for plot_file in result["plot_files"]:
            print(f"  - {plot_file}")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Check the log file for more details.")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
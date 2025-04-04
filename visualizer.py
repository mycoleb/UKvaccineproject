import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import numpy as np
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UKCovidVisualizer:
    """
    Visualizes UK COVID-19 vaccination and case data.
    Includes error handling for data validation.
    """
    def _validate_dates(self, df):
        if df['date'].max() < pd.Timestamp('2023-01-01'):
            logger.warning("Data appears to be outdated")
            return False
        return True
    
    def __init__(self):
        # Set seaborn style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def _validate_data(self, df, required_columns):
        """
        Validates the input DataFrame before visualization.
        """
        if df is None:
            logger.error("No data provided for visualization")
            return False
            
        if not isinstance(df, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame")
            return False
            
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for date column and format
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                logger.warning("Date column is not in datetime format, converting...")
                try:
                    df['date'] = pd.to_datetime(df['date'])
                except Exception as e:
                    logger.error(f"Failed to convert dates: {str(e)}")
                    return False
            
            # Log date range for debugging
            logger.info(f"Visualization date range: {df['date'].min()} to {df['date'].max()}")
            
        return True
    
    def _setup_date_axis(self, ax):
        """
        Configure date formatting for the x-axis
        """
        # Format the date axis properly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        return ax
    
    def plot_vaccination_trends(self, vaccine_df):
        """
        Plots vaccination trends over time.
        """
        try:
            required_columns = [
                'date', 
                'newPeopleVaccinatedFirstDoseByPublishDate',
                'newPeopleVaccinatedSecondDoseByPublishDate',
                'newPeopleReceivingBoosterDose'
            ]
            
            if not self._validate_data(vaccine_df, required_columns):
                return None
                
            plt.figure(figsize=(14, 8))
            
            # Plot daily vaccinations
            ax1 = plt.subplot(2, 1, 1)
            plt.plot(vaccine_df['date'], vaccine_df['newPeopleVaccinatedFirstDoseByPublishDate'], 
                    label='First Dose', color='blue')
            plt.plot(vaccine_df['date'], vaccine_df['newPeopleVaccinatedSecondDoseByPublishDate'], 
                    label='Second Dose', color='green')
            plt.plot(vaccine_df['date'], vaccine_df['newPeopleReceivingBoosterDose'], 
                    label='Booster Dose', color='purple')
            
            plt.title('Daily COVID-19 Vaccinations in the UK')
            plt.ylabel('Number of People')
            plt.legend()
            self._setup_date_axis(ax1)
            
            # Plot cumulative vaccinations
            ax2 = plt.subplot(2, 1, 2)
            plt.plot(vaccine_df['date'], vaccine_df['cumPeopleVaccinatedFirstDoseByPublishDate'], 
                    label='Cumulative First Dose', color='blue')
            plt.plot(vaccine_df['date'], vaccine_df['cumPeopleVaccinatedSecondDoseByPublishDate'], 
                    label='Cumulative Second Dose', color='green')
            
            plt.title('Cumulative COVID-19 Vaccinations in the UK')
            plt.ylabel('Number of People')
            plt.xlabel('Date')
            plt.legend()
            self._setup_date_axis(ax2)
            
            plt.tight_layout()
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting vaccination trends: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_case_trends(self, case_df):
        """
        Plots COVID-19 case and death trends over time.
        """
        
        try:
            required_columns = ['date', 'newCasesByPublishDate']
            if not self._validate_data(case_df, required_columns):
                return None
            
            # Create simplified plot if death data is missing
            plt.figure(figsize=(12, 6))
            plt.bar(case_df['date'], case_df['newCasesByPublishDate'], 
                color='orange', alpha=0.7)
            plt.title('Daily COVID-19 Cases in the UK')
                # Plot daily cases and deaths
            ax1 = plt.subplot(2, 1, 1)
                
            #label=('New Cases', color='orange', alpha=0.7)
            plt.title('Daily COVID-19 Cases in the UK')
            plt.ylabel('Number of Cases')
            self._setup_date_axis(ax1)
            
            ax2 = plt.subplot(2, 1, 2)
            plt.bar(case_df['date'], case_df['newDeaths28DaysByPublishDate'], 
                   label='New Deaths', color='red', alpha=0.7)
            plt.title('Daily COVID-19 Deaths in the UK (within 28 days of positive test)')
            plt.ylabel('Number of Deaths')
            plt.xlabel('Date')
            self._setup_date_axis(ax2)
            
            plt.tight_layout()
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting case trends: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_vaccination_vs_cases(self, vaccine_df, case_df):
        """
        Plots vaccination rates vs case rates over time.
        """
        try:
            if vaccine_df is None or case_df is None:
                logger.error("Both vaccine and case data must be provided")
                return None
                
            # Merge the two dataframes on date
            merged_df = pd.merge(vaccine_df, case_df, on='date', how='inner')
            
            # Log the dates for debugging
            logger.info(f"Merged data date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
            
            required_columns = [
                'date',
                'cumPeopleVaccinatedFirstDoseByPublishDate',
                'newCasesByPublishDate'
            ]
            
            if not self._validate_data(merged_df, required_columns):
                return None
                
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            # Plot vaccination data on primary axis
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative First Doses', color=color)
            ax1.plot(merged_df['date'], merged_df['cumPeopleVaccinatedFirstDoseByPublishDate'], 
                    color=color, label='Cumulative First Doses')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Create secondary axis for cases
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Daily New Cases', color=color)
            ax2.bar(merged_df['date'], merged_df['newCasesByPublishDate'], 
                    color=color, alpha=0.3, label='Daily New Cases')
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('UK Vaccination Progress vs COVID-19 Cases')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format date axis
            self._setup_date_axis(ax1)
            
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting vaccination vs cases: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
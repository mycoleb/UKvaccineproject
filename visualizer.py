import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from matplotlib.dates import DateFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UKCovidVisualizer:
    """
    Visualizes UK COVID-19 vaccination and case data.
    Includes error handling for data validation.
    """
    
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
            
        return True
    
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
            plt.subplot(2, 1, 1)
            plt.plot(vaccine_df['date'], vaccine_df['newPeopleVaccinatedFirstDoseByPublishDate'], 
                    label='First Dose', color='blue')
            plt.plot(vaccine_df['date'], vaccine_df['newPeopleVaccinatedSecondDoseByPublishDate'], 
                    label='Second Dose', color='green')
            plt.plot(vaccine_df['date'], vaccine_df['newPeopleReceivingBoosterDose'], 
                    label='Booster Dose', color='purple')
            
            plt.title('Daily COVID-19 Vaccinations in the UK')
            plt.ylabel('Number of People')
            plt.legend()
            plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            
            # Plot cumulative vaccinations
            plt.subplot(2, 1, 2)
            plt.plot(vaccine_df['date'], vaccine_df['cumPeopleVaccinatedFirstDoseByPublishDate'], 
                    label='Cumulative First Dose', color='blue')
            plt.plot(vaccine_df['date'], vaccine_df['cumPeopleVaccinatedSecondDoseByPublishDate'], 
                    label='Cumulative Second Dose', color='green')
            
            plt.title('Cumulative COVID-19 Vaccinations in the UK')
            plt.ylabel('Number of People')
            plt.xlabel('Date')
            plt.legend()
            plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting vaccination trends: {str(e)}")
            return None
    
    def plot_case_trends(self, case_df):
        """
        Plots COVID-19 case and death trends over time.
        """
        try:
            required_columns = [
                'date',
                'newCasesByPublishDate',
                'newDeaths28DaysByPublishDate'
            ]
            
            if not self._validate_data(case_df, required_columns):
                return None
                
            plt.figure(figsize=(14, 8))
            
            # Plot daily cases and deaths
            plt.subplot(2, 1, 1)
            plt.bar(case_df['date'], case_df['newCasesByPublishDate'], 
                   label='New Cases', color='orange', alpha=0.7)
            plt.title('Daily COVID-19 Cases in the UK')
            plt.ylabel('Number of Cases')
            plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            
            plt.subplot(2, 1, 2)
            plt.bar(case_df['date'], case_df['newDeaths28DaysByPublishDate'], 
                   label='New Deaths', color='red', alpha=0.7)
            plt.title('Daily COVID-19 Deaths in the UK (within 28 days of positive test)')
            plt.ylabel('Number of Deaths')
            plt.xlabel('Date')
            plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting case trends: {str(e)}")
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
            fig.tight_layout()
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            
            return plt
            
        except Exception as e:
            logger.error(f"Error plotting vaccination vs cases: {str(e)}")
            return None
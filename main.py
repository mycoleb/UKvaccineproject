from data_fetcher import UKCovidDataFetcher
from visualizer import UKCovidVisualizer
import logging
import os
import numpy as np
import pandas as pd  # Added missing pandas import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    
    try:
        logger.info("Starting UK COVID-19 Data Visualization Project (Multi-Source)")
        
        # Initialize data fetcher and visualizer
        fetcher = UKCovidDataFetcher()
        visualizer = UKCovidVisualizer()
        
        # Fetch data from best available sources
        logger.info("Fetching case data...")
        case_data = fetcher.get_case_data(days=365)
        
        logger.info("Fetching vaccination data...")
        vaccine_data = fetcher.get_vaccination_data(days=365)
        
        logger.info("Fetching death data...")
        death_data = fetcher.get_death_data(days=365)
        
        # Debug date formats - with better error handling
        if case_data is not None and not case_data.empty:
            logger.info(f"Case data date type: {case_data['date'].dtype}")
            logger.info(f"Case data date range: {case_data['date'].min()} to {case_data['date'].max()}")
        else:
            logger.warning("Case data is empty or None")
        
        if vaccine_data is not None and not vaccine_data.empty:
            logger.info(f"Vaccine data date type: {vaccine_data['date'].dtype}")
            logger.info(f"Vaccine data date range: {vaccine_data['date'].min()} to {vaccine_data['date'].max()}")
        else:
            logger.warning("Vaccine data is empty or None")
            
        if death_data is not None and not death_data.empty:
            logger.info(f"Death data date type: {death_data['date'].dtype}")
            logger.info(f"Death data date range: {death_data['date'].min()} to {death_data['date'].max()}")
        else:
            logger.warning("Death data is empty or None")
        
        # Ensure date columns are in datetime format
        if case_data is not None and not case_data.empty and not pd.api.types.is_datetime64_any_dtype(case_data['date']):
            logger.info("Converting case data date to datetime")
            case_data['date'] = pd.to_datetime(case_data['date'])
            
        if vaccine_data is not None and not vaccine_data.empty and not pd.api.types.is_datetime64_any_dtype(vaccine_data['date']):
            logger.info("Converting vaccine data date to datetime")
            vaccine_data['date'] = pd.to_datetime(vaccine_data['date'])
            
        if death_data is not None and not death_data.empty and not pd.api.types.is_datetime64_any_dtype(death_data['date']):
            logger.info("Converting death data date to datetime")
            death_data['date'] = pd.to_datetime(death_data['date'])
        
        # Merge case and death data if both are available
        if case_data is not None and not case_data.empty and death_data is not None and not death_data.empty:
            logger.info("Merging case and death data")
            case_data = pd.merge(case_data, death_data, on='date', how='left')
            logger.info(f"Merged data shape: {case_data.shape}")
        
        # Check if we have enough data to continue
        if (case_data is None or case_data.empty) and (vaccine_data is None or vaccine_data.empty):
            logger.error("Failed to fetch required data. Exiting.")
            return
        if case_data is not None and vaccine_data is not None:
            # Find overlapping date range
            min_date = max(case_data['date'].min(), vaccine_data['date'].min())
            max_date = min(case_data['date'].max(), vaccine_data['date'].max())
            
        if min_date > max_date:
            logger.warning("No date overlap between case and vaccine data")
            # Create visualizations
            logger.info("Creating visualizations...")
            
            # Visualization 1: Vaccination trends
            if vaccine_data is not None and not vaccine_data.empty:
                logger.info("Creating vaccination trends visualization")
                vaccine_plot = visualizer.plot_vaccination_trends(vaccine_data)
                if vaccine_plot:
                    vaccine_plot.savefig('vaccination_trends.png')
                    logger.info("Saved vaccination_trends.png")
                    vaccine_plot.close()
                else:
                    logger.error("Failed to create vaccination trends plot")
            else:
                logger.warning("Skipping vaccination trends visualization due to lack of data")
            
        # Visualization 2: Case trends
        if case_data is not None and not case_data.empty:
            logger.info("Creating case trends visualization")
            case_plot = visualizer.plot_case_trends(case_data)
            if case_plot:
                case_plot.savefig('case_trends.png')
                logger.info("Saved case_trends.png")
                case_plot.close()
            else:
                logger.error("Failed to create case trends plot")
        else:
            logger.warning("Skipping case trends visualization due to lack of data")
        
        # Visualization 3: Vaccination vs cases
        if (vaccine_data is not None and not vaccine_data.empty) and (case_data is not None and not case_data.empty):
            logger.info("Creating vaccination vs cases visualization")
            
            # Find overlapping date range
            common_dates = set(vaccine_data['date']).intersection(set(case_data['date']))
            if common_dates:
                min_date = min(common_dates)
                max_date = max(common_dates)
                logger.info(f"Using common date range: {min_date} to {max_date}")
                
                filtered_vaccine = vaccine_data[(vaccine_data['date'] >= min_date) & (vaccine_data['date'] <= max_date)]
                filtered_case = case_data[(case_data['date'] >= min_date) & (case_data['date'] <= max_date)]
                
                logger.info(f"Filtered vaccine data shape: {filtered_vaccine.shape}")
                logger.info(f"Filtered case data shape: {filtered_case.shape}")
                
                # Make sure we still have data after filtering
                if not filtered_vaccine.empty and not filtered_case.empty:
                    comparison_plot = visualizer.plot_vaccination_vs_cases(filtered_vaccine, filtered_case)
                    if comparison_plot:
                        comparison_plot.savefig('vaccination_vs_cases.png')
                        logger.info("Saved vaccination_vs_cases.png")
                        comparison_plot.close()
                    else:
                        logger.error("Failed to create vaccination vs cases plot")
                else:
                    logger.warning("Not enough data for vaccination vs cases plot after filtering")
            else:
                logger.warning("No common dates found between vaccination and case data")
        else:
            logger.warning("Skipping vaccination vs cases visualization due to lack of data")
        
        logger.info("All visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Program execution completed")
if __name__ == "__main__":
    main()
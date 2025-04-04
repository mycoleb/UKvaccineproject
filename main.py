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
        
        # Merge case and death data
        if case_data is not None and death_data is not None:
            case_data = pd.merge(case_data, death_data, on='date', how='left')
        
        if case_data is None or vaccine_data is None:
            logger.error("Failed to fetch required data. Exiting.")
            return
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # Visualization 1: Vaccination trends
        if vaccine_data is not None:
            vaccine_plot = visualizer.plot_vaccination_trends(vaccine_data)
            if vaccine_plot:
                vaccine_plot.savefig('vaccination_trends.png')
                logger.info("Saved vaccination_trends.png")
                vaccine_plot.close()
        
        # Visualization 2: Case trends
        if case_data is not None:
            case_plot = visualizer.plot_case_trends(case_data)
            if case_plot:
                case_plot.savefig('case_trends.png')
                logger.info("Saved case_trends.png")
                case_plot.close()
        
        # Visualization 3: Vaccination vs cases
        if vaccine_data is not None and case_data is not None:
            comparison_plot = visualizer.plot_vaccination_vs_cases(vaccine_data, case_data)
            if comparison_plot:
                comparison_plot.savefig('vaccination_vs_cases.png')
                logger.info("Saved vaccination_vs_cases.png")
                comparison_plot.close()
        
        logger.info("All visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())  # Add detailed traceback
    finally:
        logger.info("Program execution completed")

if __name__ == "__main__":
    main()
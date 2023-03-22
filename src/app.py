"""The main file that executes the main() method."""
from configparser import ConfigParser
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# importing own modules
from module_template.module_class import SampleClass
from estimate_leadtime.flight_histograms import EstimateLeadTime

# MY_ENV_VAR = os.getenv('MY_ENV_VAR')

# reading potential config
config = ConfigParser()
config.read("config/conf.conf")

if "AM_I_IN_A_DOCKER_CONTAINER" not in os.environ:
    load_dotenv()

def main():
    """main method that executes the whole application."""
    lead_time_estimator = EstimateLeadTime()
    lead_time_estimator.generate_flight_schedule_histograms()
    lead_time_estimator.sensitivity_analysis()


if __name__ == "__main__":
    main()

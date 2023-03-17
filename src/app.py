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

# MY_ENV_VAR = os.getenv('MY_ENV_VAR')

# reading potential config
config = ConfigParser()
config.read("config/conf.conf")

if "AM_I_IN_A_DOCKER_CONTAINER" not in os.environ:
    load_dotenv()

user_name = os.environ["USER_NAME"]
password = os.environ["USER_PASSWORD"]
my_setting = config["GENERAL"]["MY_SETTING"]

RFC_TIME = 2
LAT = 3
TOA = 2
FIRST_FLIGHT = 8
LAST_FLIGHT = 13
FLIGHT_DURATION = 2.5
OFFLOAD_RATE = 0.02
LAST_MILE = 0
CUSTOMS = 0
THRESHOLD_HOURS = 16

PEAK_HOUR = 12
STD_DEV = 6


def main():
    """main method that executes the whole application.
    """

    request_list = create_sample_requests_norm()
    lead_time_list = []
    # Loop through all sample request and determine lead time for each request
    for request in request_list:
        lead_time = determine_lead_time(request, RFC_TIME, LAT, TOA, FIRST_FLIGHT, LAST_FLIGHT, FLIGHT_DURATION, OFFLOAD_RATE, LAST_MILE, CUSTOMS)
        lead_time_list.append(lead_time)
    # Create dataframe with sample requests and lead times
    df = pd.DataFrame({'request': request_list, 'lead_time': lead_time_list})
    plot_histograms(df)

# Fuction to plot 2 subplots with histograms for lead time and hour of day
def plot_histograms(df):
    """Fuction to plot 2 subplots with histograms for lead time and hour of day.
    """
    # Calculate 90% percentile of lead times
    percentile_90 = round(df['lead_time'].quantile(0.9),2)
    # Calculate 95% percentile of lead times
    percentile_95 = round(df['lead_time'].quantile(0.95),2)
    # Calculate 98% percentile of lead times
    percentile_98 = round(df['lead_time'].quantile(0.98),2)
    # Calculate average lead time
    average_lead_time = round(df['lead_time'].mean(),2)
    # Calculate minimum lead time
    min_lead_time = round(df['lead_time'].min(),2)
    # Get percentage of requests below threshold
    percentage_below_threshold = round((df['lead_time'] <= THRESHOLD_HOURS).sum() / len(df['lead_time']) * 100,2)

    offset = 0.6
    strpos = 50

    # Create 2 subplots with histogram of lead times and histogram of hour of day of requests
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Plot histogram of lead times
    df['lead_time'].hist(bins=24, ax=ax1)
    # Add 90% percentile line
    ax1.axvline(x=percentile_90, color='g', linestyle='--')
    ax1.text(percentile_90-offset, strpos, f'90% percentile: {percentile_90} h', rotation=90)
    # Add 95% percentile line
    ax1.axvline(x=percentile_95, color='y', linestyle='--')
    ax1.text(percentile_95-offset, strpos, f'95% percentile: {percentile_95} h', rotation=90)
    # Add 98% percentile line
    ax1.axvline(x=percentile_98, color='r', linestyle='--')
    ax1.text(percentile_98-offset, strpos, f'98% percentile: {percentile_98} h', rotation=90)
    # Add average lead time line
    ax1.axvline(x=average_lead_time, color='b', linestyle='dotted')
    ax1.text(average_lead_time-offset, strpos, f'Average lead time: {average_lead_time} h', rotation=90)
    # Add minimum lead time line
    ax1.axvline(x=min_lead_time, color='b', linestyle='dotted')
    ax1.text(min_lead_time-offset, strpos, f'Minimum lead time: {min_lead_time} h', rotation=90)
    # Add text with percentage of requests below threshold
    ax1.text(0.1, 0.1, f'{percentage_below_threshold}% of requests below {THRESHOLD_HOURS} h', transform=ax1.transAxes)

    # Plot histogram of hour of day of requests
    df['request'].dt.hour.hist(bins=24, ax=ax2)
    # Add title to subplots
    fig.suptitle(f'Lead time histogram with parameters: RFC_TIME={RFC_TIME}, LAT={LAT}, TOA={TOA}, FIRST_FLIGHT={FIRST_FLIGHT}, LAST_FLIGHT={LAST_FLIGHT}, FLIGHT_DURATION={FLIGHT_DURATION} \n OFFLOAD_RATE={OFFLOAD_RATE}, LAST_MILE={LAST_MILE}, CUSTOMS={CUSTOMS}', fontsize=7)
    # Add title to lead time histogram
    ax1.set_title('Lead time histogram')
    # Add title to hour of day histogram
    ax2.set_title('Hour of day histogram (request behavior)')
    # Save figure to file
    fig.savefig("output/histograms.png")

# Function that creates normally distributed sample requests with mean 12:00 and standard deviation 4 hours
def create_sample_requests_norm():
    """Function that creates normally distributed sample requests with mean 12:00 and standard deviation 4 hours.
    """
    # Draw 2000 samples from normal distribution with mean 12:00 and standard deviation 240 minutes
    samples = np.random.normal(PEAK_HOUR*60, STD_DEV*60, 2000)
    request_list = [datetime(2023, 1, 1, 0, 0) + timedelta(minutes=x) for x in samples]
    return request_list

# Function to create list of sample requests as datetime objects for every minute of the day
def create_sample_requests():
    """Function to create list of sample requests as datetime objects for every minute of the day.
    """
    # Create list of datetime objects for every minute of the day
    request_list = [datetime(2023, 1, 1, 0, 0) + timedelta(minutes=x) for x in range(0, 1440)]
    return request_list

# Function to determine lead time depending on time of request
def determine_lead_time(dt_request, rfc_time, lat, toa, first_flight, last_flight, flight_duration, offload_rate, last_mile, customs):
    """Function to determine lead time depending on time of request.
    """
    
    # Get 00:00 of dt_request
    dt_request_start = dt_request.replace(hour=0, minute=0, second=0, microsecond=0)

    # time_of_request = datetime.now()
    # Randomly chose if unit is offloaded with a chance of OFFLOAD_RATE
    offload = np.random.choice([0, 1], p=[1-offload_rate, offload_rate])
    #offload = 0
    #print(time_of_request)
    if dt_request + timedelta(hours = (rfc_time+lat)) <= dt_request_start + timedelta(hours=first_flight) and (offload == 0):
        dt_arrival = dt_request_start + timedelta(hours=(first_flight + flight_duration + toa + customs + last_mile))
    elif dt_request + timedelta(hours = (rfc_time+lat)) <= dt_request_start + timedelta(hours=last_flight) and (offload == 0):
        dt_arrival = dt_request_start + timedelta(hours=(last_flight + flight_duration + toa + customs + last_mile))
    # Too late for last flight on that day
    elif dt_request + timedelta(hours = (rfc_time+lat)) > dt_request_start + timedelta(hours=last_flight) and (offload == 0):
        dt_arrival = dt_request_start + timedelta(hours=(24 + first_flight + flight_duration + toa + customs + last_mile))
    # If units if offloaded push it to either the later flight on that day or to the first flight next day
    elif offload == 1:
        if dt_request + timedelta(hours = (rfc_time+lat)) < dt_request_start + timedelta(hours=first_flight):
            dt_arrival = dt_request_start + timedelta(hours=(last_flight + flight_duration + toa + customs + last_mile))
        else:
            dt_arrival = dt_request_start + timedelta(hours=(24 + first_flight + flight_duration + toa + customs + last_mile))
    # Getting lead time in hours
    lead_time = (dt_arrival - dt_request).total_seconds() / 3600
    return lead_time


if __name__ == "__main__":
    main()

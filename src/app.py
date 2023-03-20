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

RFC_TIME = 1
LAT = 1.5
TOA = 1.5

FLIGHT_SCHEDULE = {
    1: 3,
    2: 8,
    3: 13,
    4: 18,
    5: 23,
}

FLIGHT_DURATION = 2.5
OFFLOAD_RATE = 0.02
LAST_MILE = 0
CUSTOMS = 0
THRESHOLD_HOURS = 16

PEAK_HOUR = 8
STD_DEV = 8


def main():
    """main method that executes the whole application.
    """

    request_list = create_sample_requests_norm()
    lead_time_list = []
    flight_no_list = []
    offload_list = []
    # Loop through all sample request and determine lead time for each request
    for request in request_list:
        lead_time, flight_no, offload = determine_lead_time_flight_schedule(request,
                                                        rfc_time=RFC_TIME,
                                                        lat=LAT,
                                                        toa=TOA,
                                                        flight_schedule=FLIGHT_SCHEDULE,
                                                        flight_duration=FLIGHT_DURATION,
                                                        offload_rate=OFFLOAD_RATE,
                                                        last_mile=LAST_MILE,
                                                        customs=CUSTOMS)
        """lead_time = determine_lead_time(request,
                                        rfc_time=RFC_TIME,
                                        lat=LAT,
                                        toa=TOA,
                                        first_flight=FIRST_FLIGHT,
                                        last_flight=LAST_FLIGHT,
                                        flight_duration=FLIGHT_DURATION,
                                        offload_rate=OFFLOAD_RATE,
                                        last_mile=LAST_MILE,
                                        customs=CUSTOMS
                                        )"""
        lead_time_list.append(lead_time)
        flight_no_list.append(flight_no)
        offload_list.append(offload)
    # Create dataframe with sample requests and lead times
    df = pd.DataFrame({'request': request_list, 'lead_time': lead_time_list, "flight_no": flight_no_list, "offload": offload_list})
    #df.to_excel("output/lead_times.xlsx", index=False)
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
    # Create figure with 3 subplots


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))
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
    # Add cumulative distribution line on second y-axis
    ax1b = ax1.twinx()
    ax1b.hist(df['lead_time'], bins=200, cumulative=True, density=True, histtype='step', color='k', linestyle='dotted', linewidth=2)

    # Plot histogram of hour of day of requests
    df['request'].dt.hour.hist(bins=24, ax=ax2)
    # Plot cumulative distribution of hour of day of requests on second y axis
    ax2b = ax2.twinx()
    # Get number of seconds since midnight for each request
    the_hours = (df['request'].dt.hour * 3600 + df['request'].dt.minute * 60 + df['request'].dt.second) / 3600
    ax2b.hist(the_hours, bins=200, cumulative=True, density=True, histtype='step', color='k', linestyle='dotted', linewidth=2)

    # String of Flights from FLIGHT_SCHEDULE
    flight_string = ""
    for flight in FLIGHT_SCHEDULE:
        flight_string += f'Flight {flight}: {FLIGHT_SCHEDULE[flight]}h, '
    # Add title to subplots
    fig.suptitle(f'Lead time histogram with parameters: RFC_TIME={RFC_TIME}, LAT={LAT}, TOA={TOA}, FLIGHT_DURATION={FLIGHT_DURATION} \n OFFLOAD_RATE={OFFLOAD_RATE}, LAST_MILE={LAST_MILE}, CUSTOMS={CUSTOMS},\n FLIGHTS={flight_string}', fontsize=12)
    # Add title to lead time histogram
    ax1.set_title('Lead time histogram')
    # Add title to hour of day histogram
    ax2.set_title('Hour of day histogram (request behavior)')
    for flight in FLIGHT_SCHEDULE:
        ax2.axvline(x=FLIGHT_SCHEDULE[flight], color='r', linestyle='--', linewidth=3)
        ax2.text(FLIGHT_SCHEDULE[flight]-0.5, 100, f'Flight {flight}', rotation=90)

        ax2.axvline(x=FLIGHT_SCHEDULE[flight]-(LAT+RFC_TIME), color='orange', linestyle='--', linewidth=1)
        ax2.text(FLIGHT_SCHEDULE[flight]-(LAT+RFC_TIME)-0.5, 50, f'Cut-off flight {flight}', rotation=90)
        #print(f"adding flight {flight}")
        #ax2.annotate(f'{flight} h', xy=(FLIGHT_SCHEDULE[flight]-(LAT+RFC_TIME), (100-FLIGHT_SCHEDULE[flight])), xytext=(FLIGHT_SCHEDULE[flight]-(LAT+RFC_TIME)-0.5, (100-FLIGHT_SCHEDULE[flight])), arrowprops=dict(arrowstyle='->'))
    # Plot bar chart of number of requests per flight in original order of flights in percent of total requests
    # Get value counts as percentage of total requests
    df['flight_no'].value_counts(normalize=True).sort_index().plot.bar(ax=ax3)

    #df['flight_no'].value_counts().sort_index().plot.bar(ax=ax3)


    # Add title to bar chart
    ax3.set_title('Number of requests per flight / Flight Usage')
    # Add red vertical lines for each flight
    
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


# Function to determine lead time based on time of request and flight schedule
def determine_lead_time_flight_schedule(dt_request, rfc_time, lat, toa, flight_schedule, flight_duration, offload_rate, last_mile, customs):
    """Function to determine lead time based on time of request and flight schedule.
    """
    # Get 00:00 of dt_request
    dt_request_start = dt_request.replace(hour=0, minute=0, second=0, microsecond=0)
    # Randomly chose if unit is offloaded with a chance of OFFLOAD_RATE
    offload = np.random.choice([0, 1], p=[1-offload_rate, offload_rate])
    flight_found = False
    # Find maximum flight number
    max_flight = max(flight_schedule.keys())
    # Making sure flights are looped through in ascending order
    for flight in range(1,max_flight+1):
        if dt_request + timedelta(hours = (rfc_time+lat)) <= dt_request_start + timedelta(hours=flight_schedule[flight]):
            dt_arrival = dt_request_start + timedelta(hours=(flight_schedule[flight] + flight_duration + toa + customs + last_mile))
            flight_found = True
            flight_no = flight
            if offload == 1:
                # Check if flight+1 is contained in flight_schedule
                if flight+1 in flight_schedule:
                    dt_arrival = dt_request_start + timedelta(hours=(flight_schedule[flight+1] + flight_duration + toa + customs + last_mile))
                else:
                    dt_arrival = dt_request_start + timedelta(hours=(24 + flight_schedule[1] + flight_duration + toa + customs + last_mile))
            break

    if flight_found == False:
        dt_arrival = dt_request_start + timedelta(hours=(24 + flight_schedule[1] + flight_duration + toa + customs + last_mile))
        # Find max key in flight_schedule
        flight_no = max(flight_schedule.keys())+1

    lead_time = (dt_arrival - dt_request).total_seconds() / 3600
    return lead_time, flight_no, offload

# Function to create FLIGHT_SCHEDULE based on given number of flights per day
def create_schedule(flights_per_day):
    hours_between_flights = 24 / flights_per_day
    flight_schedule = {}
    for i in range(1, flights_per_day+1):
        flight_schedule[i] = (i-1)*hours_between_flights

    return flight_schedule

def sensitivity_analysis():
    flight_schedules = {}
    flight_durations = []
    for i in range(1, 14):
        flight_schedules[i] = create_schedule(i)

    for i in range(1,16):
        flight_durations.append(i)

    request_list = create_sample_requests_norm()
    df_final = pd.DataFrame()
    df_percentiles = pd.DataFrame()
    for flight_schedule in flight_schedules:
        for flight_duration in flight_durations:
            lead_time_list = []
            flight_no_list = []
            offload_list = []
            for request in request_list:
                lead_time, flight_no, offload = determine_lead_time_flight_schedule(request,
                                                                rfc_time=RFC_TIME,
                                                                lat=LAT,
                                                                toa=TOA,
                                                                flight_schedule=flight_schedules[flight_schedule],
                                                                flight_duration=flight_duration,
                                                                offload_rate=OFFLOAD_RATE,
                                                                last_mile=LAST_MILE,
                                                                customs=CUSTOMS)
                
                lead_time_list.append(lead_time)
                flight_no_list.append(flight_no)
                offload_list.append(offload)
                # Create list with current value of flight_schedule for every request
                flight_schedule_list = [flight_schedule] * len(request_list)
                # Create list with current value of flight_duration for every request
                flight_duration_list = [flight_duration] * len(request_list)
    # Create dataframe with sample requests and lead times
            df = pd.DataFrame({'request': request_list,
                               'lead_time': lead_time_list,
                               "flight_no": flight_no_list,
                               "offload": offload_list,
                               "flight_schedule": flight_schedule_list,
                               "flight_duration": flight_duration_list})

            # Calculate 95% percentile of lead times
            percentile_95 = df['lead_time'].quantile(0.95)
            # Create dict with flight frequency, flight duration and 95% percentile of lead times
            percentiles = pd.DataFrame({'flight_frequency': [flight_schedule],
                                        'flight_duration': [flight_duration],
                                        'percentile_95': [percentile_95],
                                        'threshold': [THRESHOLD_HOURS]})
            # add column with 'r' if 95% percentile of lead times is above threshold
            percentiles['color'] = np.where(percentiles['percentile_95'] > percentiles['threshold'], 'orange', 'c')
            # Vertically concat percentiles into df_percentiles
            df_percentiles = pd.concat([df_percentiles, percentiles], axis=0)

            # Vertically concat df into df_final
            df_final = pd.concat([df_final, df], axis=0)

    # Plot 3d surface plot of flight frequency, flight duration and 95% percentile of lead times with matplotlib. Plot dots above threshold in red
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_percentiles['flight_frequency'], df_percentiles['flight_duration'], df_percentiles['percentile_95'], c=df_percentiles['color'])
    # Set x-ticks to 1
    ax.set_xticks(np.arange(1, 14, 1))
    ax.set_xlabel('Flight Frequency (flights per day)')
    ax.set_ylabel('Flight Duration [h]')
    ax.set_zlabel('95% Percentile Lead Time [h]')
    # Add horizontal plane at z=THRESHOLD_HOURS
    #ax.plot_trisurf(df_percentiles['flight_frequency'], df_percentiles['flight_duration'], df_percentiles['threshold'], color='grey', alpha=0.5)
    # Add text with threshold value
    ax.text(1, 1, THRESHOLD_HOURS*2, "Threshold: " + str(THRESHOLD_HOURS) + "h", color='grey')
    # Add title
    ax.set_title("Sensitivity Analysis Lead Time vs. Flight Frequency and Flight Duration \n assuming equally distributed flights throughout the day")
    # Add legend manually assigning colors to labels
    #ax.legend(['Lead Time < Threshold', 'Lead Time > Threshold'], loc='upper left')

    # Save to file
    fig.savefig("output/sensitivity_analysis_3d.png")

    return df_final, df_percentiles


if __name__ == "__main__":
    main()
    df_final, df_percentile = sensitivity_analysis()
    #df_final.to_excel("output/sensitivity_analysis.xlsx")
    #fig = px.scatter_3d(df_percentile, x='flight_frequency', y='flight_duration', z='percentile_95',
    #df_percentile.to_excel("output/sensitivity_analysis_percentiles.xlsx")


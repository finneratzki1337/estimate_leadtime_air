"""The main file that executes the main() method."""
from configparser import ConfigParser
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')

import gradio as gr


# importing own modules
from module_template.module_class import SampleClass
from estimate_leadtime.flight_histograms import EstimateLeadTime

# MY_ENV_VAR = os.getenv('MY_ENV_VAR')

# reading potential config
config = ConfigParser()
config.read("config/conf.conf")

lead_time_estimator = EstimateLeadTime()

if "AM_I_IN_A_DOCKER_CONTAINER" not in os.environ:
    load_dotenv()

def main():
    """main method that executes the whole application."""
    
    # Define gradio interface
    inputs, outputs = gr_layout()
    gr.Interface(
        create_histo_chart,
        inputs,
        outputs=["plot", "plot"],
        title="Flight Lead Time Estimator",
        description="This app estimates the lead time of a flight.",
    ).launch(share=True)


def create_histo_chart(rfc_time,
                 lat,
                 toa,
                 flight_duration,
                 offload_rate,
                 customs_time,
                 last_mile_time,
                 target_delivery_time,
                 peak_time, std_dev,
                 flight_schedule):
    """Getting the Lead Time Scatterplots."""

    # Generate dict from flight schedule
    flight_schedule_list = flight_schedule.split(",")
    flight_schedule = {}
    # length of flight schedule
    i = 1
    for flight in flight_schedule_list:
        flight_schedule[i] = float(flight)
        i += 1

    fig1 = lead_time_estimator.generate_flight_schedule_histograms(
        rfc_time = rfc_time,
        lat = lat,
        toa = toa,
        flight_schedule = flight_schedule,
        flight_duration = flight_duration,
        offload_rate = offload_rate,
        customs = customs_time,
        last_mile = last_mile_time,
        threshold = target_delivery_time,
        peak_hour=peak_time,
        std_dev=std_dev
    )

    fig2 = lead_time_estimator.sensitivity_analysis(rfc_time=rfc_time,lat=lat, toa=toa, offload_rate=offload_rate, customs=customs_time, last_mile=last_mile_time, threshold=target_delivery_time, peak_hour=peak_time, std_dev=std_dev)
    return [fig1, fig2]

def create_sensitivity_chart(rfc_time,
                 lat,
                 toa,
                 flight_duration,
                 offload_rate,
                 customs_time,
                 last_mile_time,
                 target_delivery_time,
                 peak_time, std_dev,
                 flight_schedule):
    """Getting the Lead Time Scatterplots."""

    # Generate dict from flight schedule
    flight_schedule_list = flight_schedule.split(",")
    flight_schedule = {}
    # length of flight schedule
    i = 1
    for flight in flight_schedule_list:
        flight_schedule[i] = float(flight)
        i += 1

    fig = lead_time_estimator.sensitivity_analysis(
        rfc_time = rfc_time,
        lat = lat,
        toa = toa,
        offload_rate = offload_rate,
        customs = customs_time,
        last_mile = last_mile_time,
        threshold = target_delivery_time,
        peak_hour=peak_time,
        std_dev=std_dev
    )
    return fig

def gr_layout():
    """Layout for the gradio app."""
    inputs = [
        # Float Input
        gr.components.Number(label="Ready for Carriage Time [h]", value = 2),
        gr.components.Number(label="Latest Acceptance Time (LAT) [h]", value = 1.5),
        gr.components.Number(label="Time Of Availability (TOA) [h]", value = 1.5),
        gr.components.Number(label="Flight Duration [h]", value = 2.5),
        gr.components.Number(label="Offload Rate (0 - 1)", value = 0.02),
        gr.components.Number(label="Customs Time [h]", value = 2),
        gr.components.Number(label="Last Mile Time [h]", value = 2),
        gr.components.Number(label="Target Delivery Time [h]", value = 16),
        gr.components.Slider(minimum=0, maximum=24, value=8, label="Peak Time of Order Behavior"),
        gr.components.Slider(minimum=0, maximum=12, value=8, label="Std. Deviation of Order Behavior"),
        gr.components.Textbox(lines=1, label="Flight Schedule (Full departure hours of day separated by comma)", value = "6, 12, 20"),
    ]

    # 2 Plots as outputs
    outputs = [gr.components.Plot(), gr.components.Plot()]

    return inputs, outputs



if __name__ == "__main__":
    main()

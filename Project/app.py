# !pip install shiny
from mlmodel import getMap
from shiny import App, render, ui
import matplotlib.pyplot as plt
import numpy as np

app_ui = ui.page_fluid(
    ui.h2("Crop Suggestion Map"),
    #ui.panel_title("Crop Suggestion Map"),

    ui.layout_sidebar(

      ui.panel_sidebar(
        ui.input_select("monthip", "Months After", [0,1,2,3,4,5,6,7,8,9,10,11,12]),
        style = "width : 25vw;"
      ),

      ui.panel_main(
        ui.output_plot("plot", height = "90vh"),
        style = "height : 90vh;"
      ),
    ),
    style = "align-content: center; width:100vw; height:100vh;"
)

def server(input, output, session):

    @output
    @render.plot
    def plot() -> object:
        ax = getMap(int(input.monthip()))
        return ax

app = App(app_ui, server)

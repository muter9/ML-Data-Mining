# Car Price Prediction Project

## Overview
This project is designed to predict car prices using a combination of web scraping, machine learning, and web development. It consists of three main parts:

### Part 1: Web Scraper for Car Data
In this part, a web scraper is built to mine car data from a specified website. The scraper collects essential information such as car make, model, year, engine capacity, mileage, and price. This data serves as the foundation for training the machine learning model.

### Part 2: Machine Learning Model
Using the data collected by the scraper, a machine learning model is developed to predict car prices. The model is trained on various features such as car type, year of manufacture, engine size, and more. The chosen model (Elastic Net) is fine-tuned and validated to ensure accurate predictions.

### Part 3: Flask Web Application
The final part of the project involves building a Flask web application. This application provides a user-friendly interface where users can input details about their car (such as make, model, year, and engine size) and receive a predicted price for their car. The application integrates the machine learning model to provide real-time predictions based on user input.

# Description of Government Response Dataset

## Data Sources

For the government response data: https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker#data

For the number of cases per country: https://www.worldometers.info/coronavirus/#countries

## Data Description

The following table describes each column in the dataset. 

| Row Title                            | Description                                                  | Coding                                                       |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CountryName                          | Name of country                                              | N/A                                                          |
| CountryCode                          | Abbreviated country name (e.x. Canada -> CAN)                | N/A                                                          |
| Date                                 | Date                                                         | N/A                                                          |
| C1_School closing                    | Record closings of schools and universities                  | 0 - no measures<br/>1 - recommend closing<br/>2 - require closing (only some levels or categories, eg just high school, or just public schools)<br/>3 - require closing all levels<br />Blank - no data |
| C2_Workplace closing                 | Record closings of workplaces                                | 0 - no measures<br/>1 - recommend closing (or recommend work from home)<br/>2 - require closing (or work from home) for some sectors or categories of workers<br/>3 - require closing (or work from home) for all-but-essential workplaces (eg grocery stores, doctors)<br/>Blank - no data |
| C3_Cancel public events              | Record cancelling public events                              | 0 - no measures<br/>1 - recommend cancelling<br/>2 - require cancelling<br/>Blank - no data |
| C4_Restrictions on gatherings        | Record limits on private gatherings                          | 0 - no restrictions<br/>1 - restrictions on very large gatherings (the limit is above 1000 people)<br/>2 - restrictions on gatherings between 101-1000 people<br/>3 - restrictions on gatherings between 11-100 people<br/>4 - restrictions on gatherings of 10 people or less<br/>Blank - no data |
| C5_Close public transport            | Record closing of public transport                           | 0 - no measures<br/>1 - recommend closing (or significantly reduce volume/route/means of transport available)<br/>2 - require closing (or prohibit most citizens from using it)<br/>Blank - no data |
| C6_Stay at home requirements         | Record orders to "shelter-in-place" and otherwise confine to the home | 0 - no measures<br/>1 - recommend not leaving house<br/>2 - require not leaving house with exceptions for daily exercise, grocery shopping, and 'essential' trips<br/>3 - require not leaving house with minimal exceptions (eg allowed to leave once a week, or only one person can leave at a time, etc)<br/>Blank - no data |
| C7_Restrictions on internal movement | Record restrictions on internal movement between cities/regions | 0 - no measures<br/>1 - recommend not to travel between regions/cities<br/>2 - internal movement restrictions in place Blank - no data |
| C8_International travel controls     | Record restrictions on international travel<br /><br />Note: this records policy for foreign travellers, not citizens | 0 - no restrictions<br/>1 - screening arrivals<br/>2 - quarantine arrivals from some or all regions<br/>3 - ban arrivals from some regions<br/>4 - ban on all regions or total border closure<br/>Blank - no data |
| H1_Public information campaigns      | Record presence of public info campaigns                     | 0 - no Covid-19 public information campaign<br/>1 - public officials urging caution about Covid-19<br/>2- coordinated public information campaign (eg across traditional and social media)<br/>Blank - no data |
| H2_Testing policy                    | Record government policy on who has access to testing<br /><br />Note: this records policies about testing for current infection not testing for immunity | 0 - no testing policy<br/>1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers, admitted to hospital, came into contact with a known case, returned from overseas)<br/>2 - testing of anyone showing Covid-19 symptoms<br/>3 - open public testing (eg "drive through" testing available to asymptomatic people)<br/>Blank - no data |
| H3_Contact tracing                   | Record government policy on contact tracing after a positive diagnosis<br /><br />Note: we are looking for policies that would identify all people potentially exposed to Covid-19; voluntary bluetooth apps are unlikely to achieve this | 0 - no contact tracing<br/>1 - limited contact tracing; not done for all cases<br/> 2 - comprehensive contact tracing; done for all identified cases |
| ConfirmedCases                       | Cumulative number of confirmed cases.                        | N/A                                                          |
| ConfirmedDeaths                      | Cumulative number of confirmed cases.                        | N/A                                                          |
| HealthIndexChange                    | Notes if there was a change in government response compared to the previous day. | 0 - no change in response<br/>1 - change in response         |
| ContainmentHealthIndex               | A normalized value to give the overall impression of government activity. The value is calculated using C1-C8, and H1-H3. See the [full methodology here](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md). | A higher index corresponds to more severe measures taken. A change in this value corresponds to a change in government responses. |
| Cases after 1 week                   |                                                              | -1 - decrease in cases<br/>1 - increase in cases             |
| Cases after 2 weeks                  |                                                              | -1 - decrease in cases<br/>1 - increase in cases             |
| Cases after 3 weeks                  |                                                              | -1 - decrease in cases<br/>1 - increase in cases             |
| Cases after 4 weeks                  |                                                              | -1 - decrease in cases<br/>1 - increase in cases             |

### Input Features

Experimentation must be performed to see which set of input features should be used. Regardless of implementation the following features will be used:

* C1_School closing
* C2_Workplace closing
* C3_Cancel public events
* C4_Restrictions on gatherings
* C5_Close public transport
* C6_Stay at home requirements
* C7_Restrictions on internal movement
* C8_International travel controls
* H1_Public information campaigns
* H2_Testing policy
* H3_Contact tracing

Additionally, 'CountryCode' can be used as in input feature if we want to limit the predications for specific countries. We may want to do this if we find via experimentation that the model is not accurate. 

Having 'CountryCode' as an input feature would most likely result in a decision tree using 'CountryCode' as the feature to split on the root note. This would most likely result in the model more accurately predicting a rise/fall in cases for a specific country. The reason this may be needed is that there are likely factors not taken into account that may result in the cases increasing/decreasing differently even if two countries have similar response measures. These factors can be cultural, political, geographical, etc. Based on what's been seen in the raw data it doesn't seem likely that we'll have to do this, but we'll see.

### Target Features

Experimentation will be required to see **one** which of the following target features will be used:

* Cases after 1 week
* Cases after 2 weeks
* Cases after 3 weeks
* Cases after 4 weeks

Whichever of the above features results in the most accurate prediction is what should be used. 

## Cleaning Steps

1. Remove unwanted countries from the [Government Response Dataset](https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker#data). 
2. Remove any columns that are not important to our problem. We kept all columns that could be used as input features as well as any columns that can provide helpful information to someone looking though the dataset.
3. Remove any rows from the beginning of a country's data if there are no cases recorded.
4. Remove any rows from the end of a country's data if there is any missing information.
5. Add a column "HealthIndexChange" and fill each cell with 1 if there was a change in "ContainmentHealthIndex" from the previous day, otherwise a 0.
6. Add the four columns ''Cases after 1 week", "Cases after 2 weeks", "Cases after 3 weeks", "Cases after 4 weeks". 
7. For every row there is a 1 in the "HealthIndexChange" decide if there should be a 1 or -1 in the "Cases after x week(s)" columns depending on if there is a rise or fall in cases. To do so, go [here](https://www.worldometers.info/coronavirus/#countries) and navigate to "Countries", then select the country in question. Scrolling down you will find a plot of "Active Cases in \<Country\>". This plot can be used to see the rise/fall in countries on any given day.


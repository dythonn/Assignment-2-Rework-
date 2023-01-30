import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


import pandas as pd

def read_world_bank_data(filename):
    """
    Reads a dataframe in World-bank format and returns two dataframes:
    one with years as columns and one with countries as columns.
    
    Parameters:
    filename (str): filename of the dataframe to be read
    
    Returns:
    Tuple: Two dataframes, one with years as columns and one with countries as columns.
    """
    # read the dataframe from the file
    df = pd.read_csv(filename)
    
    # get the list of years
    year_list = list(df.iloc[:, 4:].columns)
    
    # create a dataframe with the list of years
    year_df = pd.DataFrame(year_list)
    
    # melt the dataframe to create a dataframe with countries as columns
    df_by_country = pd.melt(df, id_vars=['Country Name', 'Series Name', 'Series Code', 'Country Code'], value_vars=year_list)
    
    # clean the 'variable' column
    df_by_country['variable'] = df_by_country['variable'].str[0:4]
    
    # rename the 'variable' column to 'Year' and 'value' column to 'Values in %'
    df_by_country.rename({'variable': 'Year', 'value': 'Values in %'}, axis=1, inplace=True)
    
    # change the data type of the 'Year' column from object to datetime format
    df_by_country['Year'] = pd.to_datetime(df_by_country['Year'])
    
    # transpose the dataframe with countries as columns to get a dataframe with years as columns
    df_by_year = df_by_country.pivot_table(index=['Country Name', 'Series Name', 'Series Code', 'Country Code'], columns='Year', values='Values in %')
    
    return year_df, df_by_country



# Call the function to read the datafile
year_df, df_by_country = read_world_bank_data('final dataset.csv')
print(year_df)
print(df_by_country.head())



# =============================================================================
#                             Pre-processing
# =============================================================================


# print (df_by_country.info())

def process_data(df_by_country):
    df_by_country = df_by_country.replace('..', np.nan)
    df_by_country['Values in %'] = df_by_country['Values in %'].astype(float)
    df_by_country = df_by_country.dropna()
    df_by_country['Year'] = df_by_country['Year'].dt.year
    df_dup = df_by_country
    pak = df_dup[df_dup["Country Code"] == 'PAK']
    ind = df_dup[df_dup["Country Code"] == 'IND']
    usa = df_dup[df_dup["Country Code"] == 'USA']
    return df_by_country, df_dup, pak, ind, usa


df_by_country, df_dup, pak, ind, usa = process_data(df_by_country)


""" Indicators For LinePlot """

def get_pak_indicators(pak):
    pak_indicator = pak[(pak["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
    pak_indicator1 = pak[ (pak["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    pak_indicator2 = pak[(pak["Series Code"] == 'SL.UEM.ADVN.ZS')]
    return pak_indicator, pak_indicator1, pak_indicator2

pak_indicator, pak_indicator1, pak_indicator2 = get_pak_indicators(pak)

def get_ind_indicators(ind):
    ind_indicator = ind[(ind["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
    ind_indicator1 = ind[ (ind["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    ind_indicator2 = ind[(ind["Series Code"] == 'SL.UEM.ADVN.ZS')]
    return ind_indicator, ind_indicator1, ind_indicator2

ind_indicator, ind_indicator1, ind_indicator2 = get_ind_indicators(ind)


def get_usa_indicators(usa):

    usa_indicator = usa[(usa["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
    usa_indicator1 = usa[(usa["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
    usa_indicator2 = usa[(usa["Series Code"] == 'SL.TLF.ADVN.MA.ZS')]
    
    return usa_indicator, usa_indicator1, usa_indicator2

usa_indicator, usa_indicator1, usa_indicator2 = get_usa_indicators(usa)



""" Indicators For BarPlot """


def get_pak_bar_indicators(pak):
    
    """The function get_usa_pak_indicators takes the input pak, a dataframe, and returns four dataframes containing data on four education-related indicators for the Pak. The indicators include:
        Master's degree or equivalent, population 25+ years, male
        Master's degree or equivalent, population 25+ years, female
        Unemployment with advanced education, male
        Unemployment with advanced education, female
        The returned dataframes can then be used for further analysis and visualization."""
    
    bar_pak_indicator = pak[(pak["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
    bar_pak_indicator1 = pak[ (pak["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
    bar_pak_indicator2 = pak[(pak["Series Code"] == 'SL.TLF.ADVN.MA.ZS')]
    bar_pak_indicator3 = pak[(pak["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    return bar_pak_indicator, bar_pak_indicator1, bar_pak_indicator2, bar_pak_indicator3

bar_pak_indicator, bar_pak_indicator1, bar_pak_indicator2, bar_pak_indicator3 = get_pak_bar_indicators(pak)


def get_ind_bar_indicators(ind):
    
    """The function get_usa_bar_indicators takes the input ind, a dataframe, and returns four dataframes containing data on four education-related indicators for the IND. The indicators include:
        Master's degree or equivalent, population 25+ years, male
        Master's degree or equivalent, population 25+ years, female
        Unemployment with advanced education, male
        Unemployment with advanced education, female
        The returned dataframes can then be used for further analysis and visualization."""

    bar_ind_indicator = ind[(ind["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
    bar_ind_indicator1 = ind[ (ind["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
    bar_ind_indicator2 = ind[(ind["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
    bar_ind_indicator3 = ind[(ind["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    return bar_ind_indicator, bar_ind_indicator1, bar_ind_indicator2, bar_ind_indicator3

bar_ind_indicator, bar_ind_indicator1, bar_ind_indicator2, bar_ind_indicator3 = get_ind_bar_indicators(ind)


def get_usa_bar_indicators(ind):
    """The function get_usa_bar_indicators takes the input usa, a dataframe, and returns four dataframes containing data on four education-related indicators for the USA. The indicators include:
        Master's degree or equivalent, population 25+ years, male
        Master's degree or equivalent, population 25+ years, female
        Unemployment with advanced education, male
        Unemployment with advanced education, female
        The returned dataframes can then be used for further analysis and visualization."""
    
    bar_usa_indicator = usa[(usa["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
    bar_usa_indicator1 = usa[ (usa["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
    bar_usa_indicator2 = usa[(usa["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
    bar_usa_indicator3 = usa[(usa["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    return bar_usa_indicator, bar_usa_indicator1, bar_usa_indicator2, bar_usa_indicator3

bar_usa_indicator, bar_usa_indicator1, bar_usa_indicator2, bar_usa_indicator3 = get_usa_bar_indicators(ind)
    


def country_counter(con, sc):
    """Takes a Country and indicator and pass their value"""
    
    country = df_dup[df_dup["Country Code"] == con]
    indicator = country [(country ["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
    indicator1 = country [(country["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    indicator2 = country [(country ["Series Code"] == 'SE.XPD.TOTL.GB.ZS')]
    
    return indicator,indicator1,indicator2


print (ind_indicator, ind_indicator1, ind_indicator2)
print (pak_indicator, pak_indicator1, pak_indicator2)
print (usa_indicator, usa_indicator1, usa_indicator2)

# =============================================================================
#                            Statistical Comparison
# =============================================================================


def compare_indicator_stats(df, countries, indicators):
    
    """
    The function takes in a dataframe (df), a list of countries (countries), and a list of indicators (indicators)
    and returns a dictionary of country statistics for each country and indicator.
    
    The statistics include the mean, median, and standard deviation of the 'Values in %' for each indicator.
    
    Args:
    df (pd.DataFrame): A dataframe containing information about countries and indicators
    countries (list): A list of country codes to compare the statistics for
    indicators (list): A list of indicator codes to compare the statistics for
    
    Returns:
    dict: A dictionary with country codes as keys and a nested dictionary of indicator statistics as values.
    """
    
    country_stats = {}
    for country in countries:
        country_df = df[df["Country Code"] == country]
        country_indicators = {}
        for indicator in indicators:
            indicator_df = country_df[country_df["Series Code"] == indicator]
            indicator_mean = np.mean(indicator_df["Values in %"])
            indicator_median = np.median(indicator_df["Values in %"])
            indicator_std = np.std(indicator_df["Values in %"])
            country_indicators[indicator] = {"mean": indicator_mean,
                                             "median": indicator_median,
                                             "std": indicator_std}
        country_stats[country] = country_indicators
        
    return country_stats

countries = ['PAK',"","", 'IND',"","",'USA']
indicators = ['SE.PRM.GINT.MA.ZS', 'SE.PRM.GINT.FE.ZS', 'SL.UEM.ADVN.ZS']

country_stats = compare_indicator_stats(df_by_country, countries, indicators)
print (country_stats)

# =============================================================================
#                             Line Graph
# =============================================================================


def line_graph(country):
    """
    This function generates a line plot graph of three education indicators for the selected country over the years.

    The function takes in a single argument country, which is a string specifying the country to be plotted.
    The possible values of country are "PAK" for Pakistan, "IND" for India, and "USA" for United States of America.

    The graph shows the trends of three education indicators over the years for the selected country. The three indicators are:

    Gross intake ratio in first grade of primary education, male
    Gross intake ratio in first grade of primary education, female
    Government expenditure on education, total (% of government expenditure)
    The x-axis represents the years, and the y-axis represents the percentage.

    The function saves the plot as an image file with the name "linePlotLanguages.png" in the current working directory and displays the plot on the screen.
    """
    
    plt.figure(figsize=(11, 8))
    
    
    # takes x and y as inputs to the plot
    one_indi_yr = pak_indicator['Year']
    one_indi_ = pak_indicator['Values in %']
    two_indi_yr = pak_indicator1['Year']
    two_indi_ = pak_indicator1['Values in %']
    three_indi_yr = pak_indicator2['Year']
    three_indi_ = pak_indicator2['Values in %']
    
    one_indi_yr_z = ind_indicator['Year']
    one_indi_z = ind_indicator['Values in %']
    two_indi_yr_z = ind_indicator1['Year']
    two_indi_z = ind_indicator1['Values in %']
    three_indi_yr_z = ind_indicator2['Year']
    three_indi_z = ind_indicator2['Values in %']
    
    one_indi_yr_u = usa_indicator['Year']
    one_indi_u = usa_indicator['Values in %']
    two_indi_yr_u = usa_indicator1['Year']
    two_indi_u = usa_indicator1['Values in %']
    three_indi_yr_u = usa_indicator2['Year']
    three_indi_u = usa_indicator2['Values in %']
    
    if(country=="PAK"):
        plt.plot(one_indi_yr, one_indi_, linewidth = 1.2)
        plt.plot(two_indi_yr, two_indi_, linewidth = 1.2)
        plt.plot(three_indi_yr, three_indi_, linewidth = 1.2)
        plt.title('Pakistan', fontsize = 18)
        plt.legend(['Gross intake ratio in first grade of primary education, male', 'Gross intake ratio in first grade of primary education, female', 'Government expenditure on education, total (% of government expenditure)'], loc='center', fontsize = 15.5, frameon = False)
        
    elif(country=="IND"):
        plt.plot(one_indi_yr_z, one_indi_z, linewidth = 1.2)
        plt.plot(two_indi_yr_z, two_indi_z, linewidth = 1.2)
        plt.plot(three_indi_yr_z, three_indi_z, linewidth = 1.2)
        plt.title('India', fontsize = 18)
        plt.legend(['Gross intake ratio in first grade of primary education, male', 'Gross intake ratio in first grade of primary education, female', 'Government expenditure on education, total (% of government expenditure)'], loc='center', fontsize = 15.5, frameon = False)
        
    elif(country=="USA"):
        plt.plot(one_indi_yr_u, one_indi_u, linewidth = 1.2)
        plt.plot(two_indi_yr_u, two_indi_u, linewidth = 1.2)
        plt.plot(three_indi_yr_u, three_indi_u, linewidth = 1.2) 
        plt.title('United States of America', fontsize = 18)
        plt.legend(['Gross intake ratio in first grade of primary education, male', 'Gross intake ratio in first grade of primary education, female', 'Government expenditure on education, total (% of government expenditure)'], loc='center', fontsize = 15.5, frameon = False)
            
    plt.xlabel('Years', fontsize = 17)
    plt.ylabel('Percentage', fontsize = 17)
    
    
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.tight_layout()
    
    plt.savefig('linePlotLanguages.png')
    plt.show()
    

line_graph("IND")
line_graph("USA")
line_graph("PAK")



# =============================================================================
#                               Bar Plot
# =============================================================================


def bar_plot(value_in_per):
    """ Takes a percentage values data and plot with respect to countries
                    and returns the statistics/values """

    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.22       # the width of the bars

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Data for each country
    yvals = [pak_indicator.iloc[2]['Values in %'], pak_indicator.iloc[3]['Values in %'],
             pak_indicator.iloc[4]['Values in %'], pak_indicator.iloc[5]['Values in %']]
    zvals = [ind_indicator.iloc[2]['Values in %'], ind_indicator.iloc[3]['Values in %'],
             ind_indicator.iloc[4]['Values in %'], ind_indicator.iloc[5]['Values in %']]
    kvals = [usa_indicator.iloc[2]['Values in %'], usa_indicator.iloc[3]['Values in %'],
             usa_indicator.iloc[4]['Values in %'], usa_indicator.iloc[5]['Values in %']]
    
    # Plotting the bars
    rects1 = ax.bar(ind, yvals, width, color='r', label='Pakistan')
    rects2 = ax.bar(ind + width, zvals, width, color='g', label='India')
    rects3 = ax.bar(ind + width * 2, kvals, width, color='b', label='USA')
    
    # Adding title and labels
    ax.set_title("Unemployment with Advanced Education", fontsize=18)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('2014', '2016', '2017', '2018'), fontsize=12)
    ax.legend(fontsize=12)
    
    # Adding values on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom', fontsize=12)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.show()
    

bar_plot("")


# =============================================================================
#                             Heatmap Correlation
# =============================================================================

def heatmap_bar(name):
    """Takes Values in % of the indicators data and compares the correlation"""
    
    cc=pd.DataFrame()
    if(name=="PAK"):
        cc["Master's or equivalent, population 25+, male"]=bar_pak_indicator['Values in %'][0:7].values
        cc["Master's or equivalent, population 25+, female"]=bar_pak_indicator1['Values in %'][0:7].values
        cc["Unemployment with advanced education"]=bar_pak_indicator2['Values in %'][0:7].values
        corr = cc.corr()
        title = "Pakistan Correlation Plot"
    elif(name=="IND"):
        cc["Master's or equivalent, population 25+, male"]=bar_ind_indicator['Values in %'][0:6].values
        cc["Master's or equivalent, population 25+, female"]=bar_ind_indicator1['Values in %'][0:6].values
        cc["Unemployment with advanced education"]=bar_ind_indicator2['Values in %'][0:6].values
        corr = cc.corr()
        title = "India Correlation Plot"
    elif(name=="USA"):
        cc["Master's or equivalent, population 25+, male"]=bar_usa_indicator['Values in %'][0:6].values
        cc["Master's or equivalent, population 25+, female"]=bar_usa_indicator1['Values in %'][0:6].values
        cc["Unemployment with advanced education"]=bar_usa_indicator2['Values in %'][0:6].values
        corr = cc.corr()
        title = "United States Correlation Plot"

    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap='tab20')
    

    # Show all ticks and labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)

    # Add values inside the squares
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, '%.2f' % corr.iloc[i, j], ha="center", va="center", color="white")
    plt.title(title)        
    plt.show()

heatmap_bar("PAK")
heatmap_bar("IND")
heatmap_bar("USA")



import joblib
import datetime
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, Easter
from pandas.tseries.offsets import Day, CustomBusinessDay

class FrenchJoursFeries(AbstractHolidayCalendar):
    """ Custom Holiday calendar for France based on
        https://en.wikipedia.org/wiki/Public_holidays_in_France
      - 1 January: New Year's Day
      - Moveable: Easter Monday (Monday after Easter Sunday)
      - 1 May: Labour Day
      - 8 May: Victory in Europe Day
      - Moveable Ascension Day (Thursday, 39 days after Easter Sunday)
      - 14 July: Bastille Day
      - 15 August: Assumption of Mary to Heaven
      - 1 November: All Saints' Day
      - 11 November: Armistice Day
      - 25 December: Christmas Day
    """
    rules = [
        Holiday('New Years Day', month=1, day=1),
        EasterMonday,
        Holiday('Labour Day', month=5, day=1),
        Holiday('Victory in Europe Day', month=5, day=8),
        Holiday('Ascension Day', month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday('Bastille Day', month=7, day=14),
        Holiday('Assumption of Mary to Heaven', month=8, day=15),
        Holiday('All Saints Day', month=11, day=1),
        Holiday('Armistice Day', month=11, day=11),
        Holiday('Christmas Day', month=12, day=25)
    ]

def date_conversion(date, holidays):#convertir heures à prédire en features

    date_time_obj = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
    df = pd.DataFrame({'date': [date_time_obj]})
    df['heure'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['jour_semaine'] = df['date'].dt.dayofweek
    df['trimestre'] = df['date'].dt.quarter
    df['mois'] = df['date'].dt.month
    df['annee'] = df['date'].dt.year
    df['jour_annee'] = df['date'].dt.dayofyear
    df['jour_mois'] = df['date'].dt.day
    df['semaine'] = df['date'].dt.week
    seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    month_to_season = dict(zip(range(1,13), seasons))
    df['saison'] = df['date'].dt.month.map(month_to_season)
    df['jour_ferie'] = df.index.isin(holidays)
    df['jour_ferie'] = df['jour_ferie'].astype(int)
    bins = [0,6,11,13,19,23]#0-6h nuit/7-12h matin/12-14h pause dejeuner/14-20h apres midi/20-24h soir 
    labels = [0,1,2,3,4]#nuit/matin/pausedejeuner/apresmidi/soir
    df['plage_horaire'] = pd.cut(df['heure'], bins=bins, labels=labels, include_lowest=True)#to include 0
    df['plage_horaire'] = df['plage_horaire'].astype(int)

    df = df.loc[~((df['heure'] < 8) | (df['heure'] > 19))]

    X_pred = df[['heure', 'jour_semaine', 'minute','trimestre','mois','annee','jour_annee', 'jour_mois', 'semaine','saison', 'jour_ferie', 'plage_horaire']]
                
    return X_pred


def predict():


    model = joblib.load("model.pkl")

    pred_date =  "01-01-2021"

    cal = FrenchJoursFeries()

    holidays = cal.holidays()

    pred_date = datetime.datetime.strptime(pred_date, '%d-%m-%Y')

    day_to_predict = str(pred_date.date()) + "T08:00:00" #start at 8 o'clocks

    day_to_predict = datetime.datetime.strptime(day_to_predict, "%Y-%m-%dT%H:%M:%S")#back to datetime format

    nbr_quarter_hour = 45 #cf nbr of quarter of hour (15 min steps, from 8AM -7PM)

    hour_list_timestamp = [day_to_predict + datetime.timedelta(minutes=15*x) for x in range(nbr_quarter_hour)]

    hour_list = []

    for i in hour_list_timestamp:
        hour_formated = i.strftime("%Y-%m-%dT%H:%M:%S")
        hour_list.append(hour_formated)

    X_pred = pd.DataFrame(columns=[['heure', 'jour_semaine', 'minute','trimestre','mois','annee','jour_annee', 'jour_mois', 'semaine','saison', 'jour_ferie', 'plage_horaire']])
    i = 0

    for hour in hour_list:

        df_encoded_hours = date_conversion(hour, holidays)
        encoded_hours = df_encoded_hours.values
        encoded_hours = encoded_hours.flatten()#2D to 1D array
        X_pred.loc[i] = encoded_hours
        i += 1

    X_pred = X_pred.astype(int)

    df_predictions = pd.DataFrame()
    df_predictions['time'] = hour_list
    df_predictions['predictions'] = model.predict(X_pred)
    df_predictions['predictions'] = df_predictions['predictions'].astype(int)

    df_predictions.to_csv (r'export_dataframe.csv', index = False, header=True)




if __name__ == '__main__':
    predict()
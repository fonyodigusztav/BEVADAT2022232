import pandas as pd

class NJCleaner:
    
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
    
    def order_by_scheduled_time(self):
        self.data = self.data.sort_values(by=['scheduled_time'])
        return self.data
    
    def drop_columns_and_nan(self):
        self.data = self.data.dropna()
        self.data = self.data.drop(['from', 'to'], axis=1)
        return self.data
    
    def convert_date_to_day(self):
        self.data['day'] = pd.to_datetime(self.data['date']).dt.day_name()
        self.data = self.data.drop(['date'], axis=1)
        return self.data
    
    def convert_scheduled_time_to_part_of_the_day(self):
        def get_part_of_day(hour):
            if 4 <= hour < 8:
                return 'early_morning'
            elif 8 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 16:
                return 'afternoon'
            elif 16 <= hour < 20:
                return 'evening'
            elif 20 <= hour <= 23:
                return 'night'
            else:
                return 'late_night'
        
        self.data['part_of_the_day'] = pd.to_datetime(self.data['scheduled_time']).dt.hour.apply(get_part_of_day)
        self.data = self.data.drop(['scheduled_time'], axis=1)
        return self.data
    
    def convert_delay(self):
        self.data['delay'] = self.data['delay_minutes'].apply(lambda x: 1 if x >= 5 else 0)
        return self.data
    
    def drop_unnecessary_columns(self):
        self.data = self.data.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1)
        return self.data
    
    def save_first_60k(self, path):
        self.data.iloc[:60000].to_csv(path, index=False)
    
    def prep_df(self, path='data/NJ.csv'):
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        self.save_first_60k(path)
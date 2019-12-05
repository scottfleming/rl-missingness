from datetime import datetime

def one_hot_encode(self, df, columns_to_one_hot):
    import category_encoders as ce
    import re

    one_hot_encoder = ce.OneHotEncoder(cols=columns_to_one_hot, use_cat_names=True)
    one_hot_encoder.fit(df)
    df = one_hot_encoder.transform(df)

    # The category_encoders package creates extra empty columns
    # for some reason, so we just delete them in the end
    for col in df.columns:
        if re.search('_-1', col):
            df.drop(col, axis=1, inplace=True)

    return df


def hours_between(self, t1_str, t2_str, str_format='%Y-%m-%d %H:%M:%S'):
    t1 = datetime.strptime(t1_str, str_format)
    t2 = datetime.strptime(t2_str, str_format)
    time_diff_seconds = abs((t2 - t1).total_seconds())
    time_diff_hrs = time_diff_seconds / 60. / 60.
    return time_diff_hrs
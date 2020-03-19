from ladvien_ml import FeaturePrep

fp = FeaturePrep()

test = ['neat_feature.0.0',
        'different_feature.1.0',
        'empty_feature.nan',
        'one_value_test'
]

def make_ohe_feature_readable(feature_name, seperator = '.', print_output = False, separate_value = False):
        """

                feature_name: str, name of the feature.
                print_output: Boolean, print the processed feature name.
                separate_value: Boolean, if true, returns a tuple containing the feature name and the value.

                When one-hot encoding (OHE) values the column names often get garbled.  This method will take a
                OHE string and return a human readable name.  The string should be of the type

                name_of_feature_value, e.g., 'weight_range_150'

                >>> print(make_ohe_feature_readable(weight_range_150))
                "Weight Range = 150"
        """

        # When avoiding the the dummy variable trap two category variables
        # will not have a seperator or value in the feature name.
        if seperator not in feature_name:
                feature_name += f'{seperator}1'

        # If underscores are used instead of spaces, replace them.
        feature_name = feature_name.replace('_', ' ')

        feature_name = feature_name.split(seperator)
        value = feature_name[-1]

        # Use SQL like nulls.
        if value == 'nan':
                value = 'NULL'
        
        # If the dummy value was encoded as a float, clean it up.
        if '.0' in value:
                value = value.replace('.0', '')

        # Put the human readable name back together.
        feature_name = feature_name[0:-1]
        feature_name = ' '.join(feature_name).title()
        full_output = f'{feature_name} = {value}'

        if print_output:
            print(full_output)

        if separate_value:
            return (feature_name, value)
        else:
            return full_output


for feature in test:
    make_ohe_feature_readable(feature, seperator = '.', print_output = True)
    

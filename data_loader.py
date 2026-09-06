import pandas as pd
from io import BytesIO


def load_dataset(uploaded_file):

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                BytesIO(
                    uploaded_file.getvalue()
                )
            )

        return df, None

    except Exception as e:

        return None, e
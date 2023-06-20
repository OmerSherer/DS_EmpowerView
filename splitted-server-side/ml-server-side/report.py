def insert_confidences_to_tables(df_confidence, interviewId, executeQuery):
    df_frequency = df_confidence.drop(["timestamp", "label"], axis=1)
    columns = df_frequency.columns
    counts = {col: 0 for col in columns}
    df_labels = df_confidence["label"]
    for label in df_labels.values:
        counts[label] += 1
    frequency = [counts[column] / df_confidence.shape[0] for column in columns]

    executeQuery(
        f"""CREATE TABLE IF NOT EXISTS ConfidencesByTime_{interviewId}
                (timestamp FLOAT, label varchar(10), angry FLOAT, bored FLOAT, disgust FLOAT, happy FLOAT, sad FLOAT, shy FLOAT, stressed FLOAT, surprised FLOAT)"""
    )

    def processRow(iterrow):
        _, row = iterrow
        timestamp = row["timestamp"] / 1000
        label = row["label"]
        angry_confidence = row["angry"]
        bored_confidence = row["bored"]
        disgust_confidence = row["disgust"]
        happy_confidence = row["happy"]
        sad_confidence = row["sad"]
        shy_confidence = row["shy"]
        stressed_confidence = row["stressed"]
        surprised_confidence = row["surprised"]
        return (
            timestamp,
            label,
            angry_confidence,
            bored_confidence,
            disgust_confidence,
            happy_confidence,
            sad_confidence,
            shy_confidence,
            stressed_confidence,
            surprised_confidence,
        )

    # for _, row in df_confidence.iterrows():
    #     timestamp = row['timestamp']/1000
    #     label = row['label']
    #     angry_confidence = row['angry']
    #     bored_confidence = row['bored']
    #     disgust_confidence = row['disgust']
    #     happy_confidence = row['happy']
    #     sad_confidence = row['sad']
    #     shy_confidence = row['shy']
    #     stressed_confidence = row['stressed']
    #     surprised_confidence = row['surprised']

    executeQuery(
        f"""INSERT INTO ConfidencesByTime_{interviewId} (timestamp, label, angry, bored, disgust, happy, sad, shy, stressed, surprised)
        VALUES {tuple(map(processRow, df_confidence.iterrows())).__str__()[1:-1]}"""
    )
    # executeQuery(
    #     f"""INSERT INTO ConfidencesByTime_{interviewId} (timestamp, label, angry, bored, disgust, happy, sad, shy, stressed, surprised)
    #     VALUES ({timestamp}, '{label}', {angry_confidence}, {bored_confidence}, {disgust_confidence}, {happy_confidence},
    #         {sad_confidence}, {shy_confidence}, {stressed_confidence}, {surprised_confidence})"""

    # )

    executeQuery(
        f"""UPDATE Reports 
           SET angrypercent = {frequency[0]}, boredpercent = {frequency[1]}, disgustpercent = {frequency[2]}, happypercent = {frequency[3]}, sadpercent = {frequency[4]}, shypercent = {frequency[5]}, stressedpercent = {frequency[6]}, surprisedpercent = {frequency[7]}
           WHERE id = '{interviewId}'"""
    )

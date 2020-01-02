import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from tqdm import tqdm
from collections import Counter
from util import get_logger

logger = get_logger()


def create_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    全ての特徴を生成する。mainからはこれだけ呼べば良くする。
    """
    type_features = _type_feature(df)
    event_code_features = _event_code_feature(df)
    title_features = _title_feature(df)
    datetime_features = _datetime_feature(df)
    game_time_features = _game_time_feature(df)
    attempt_features = _attempt_features(df)

    features = pd.concat([
        type_features,
        event_code_features,
        title_features,
        datetime_features,
        game_time_features,
        attempt_features
    ], axis=1)
    logger.info('Create feature finished')
    logger.info(features.head())
    logger.info('Shape of feature: {}'.format(features.shape))
    return features


def _type_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    typeに関する特徴（ex. Assessment, Clipなど）
    """
    logger.info('Create type features')
    count_features = df.groupby(['game_session', 'type'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'type_'+col+'_counts' for col in count_features.columns}, inplace=True)
    type_features = pd.concat([count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return type_features


def _event_code_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    event_codeに関する特徴
    """
    logger.info('Create event_code features')
    count_features = df.groupby(['game_session', 'event_code'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'event_code_'+str(col)+'_counts' for col in count_features.columns}, inplace=True)
    event_code_features = pd.concat([count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return event_code_features


def _title_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    titleに関する特徴（ex. Sandcastle Builder (Activity)など）
    """
    logger.info('Create title features')
    count_features = df.groupby(['game_session', 'title'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'title_'+col+'_counts' for col in count_features.columns}, inplace=True)
    title_features = pd.concat([count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return title_features


def _datetime_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    timestampに関する特徴
    """
    logger.info('Create datetime features')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek

    hour_count_features = df.groupby(['game_session', 'hour'])['event_id'].count().unstack().fillna(0)
    hour_count_features.rename(columns={col: 'hour_'+str(col)+'_counts' for col in hour_count_features.columns}, inplace=True)
    weekday_count_features = df.groupby(['game_session', 'weekday'])['event_id'].count().unstack().fillna(0)
    weekday_count_features.rename(columns={col: 'weekday_'+str(col)+'_counts' for col in weekday_count_features.columns}, inplace=True)
    datetime_features = pd.concat([hour_count_features, weekday_count_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return datetime_features


def _game_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    game_timeに関する特徴
    """
    logger.info('Create gametime features')
    agg_features = df.groupby(['game_session'])['game_time'].agg(['sum', 'mean'])
    agg_features.rename(columns={col: 'game_time_'+col for col in agg_features.columns}, inplace=True)
    game_time_features = pd.concat([agg_features], axis=1)  # 後々コメントアウトで必要な特徴を調整できるように
    return game_time_features


def _attempt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assessmentの試行に関する特徴
    """
    logger.info('Creaet attempt features')
    all_attempts = df.loc[
        (df.type == "Assessment") & (df.title == 'Bird Measurer (Assessment)') & (df.event_code == 4110) |
        (df.type == "Assessment") & (df.title != 'Bird Measurer (Assessment)') & (df.event_code == 4100)
    ]
    all_attempts['pass_assessment'] = all_attempts['event_data'].str.contains('true')
    count_features = all_attempts.groupby(['game_session', 'pass_assessment'])['event_id'].count().unstack().fillna(0)
    count_features.rename(columns={col: 'assignment_'+str(col)+'_counts' for col in count_features.columns}, inplace=True)
    attempt_features = pd.concat([count_features]).fillna(0)  # 後々コメントアウトで必要な特徴を調整できるように
    return attempt_features


def encode_title(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
    # return train, test


def get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
            activities_labels, assess_titles, list_of_event_id, all_title_event_code, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())

            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter

        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments


def get_train_and_test(train, test,
                        win_code, list_of_user_activities, list_of_event_code,
                        activities_labels, assess_titles, list_of_event_id, all_title_event_code):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        # user_sampleに1つのins_idの全てのデータが入っている
        compiled_train += get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                activities_labels, assess_titles, list_of_event_id, all_title_event_code)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                activities_labels, assess_titles, list_of_event_id, all_title_event_code, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    return reduce_train, reduce_test


def preprocess(reduce_train, reduce_test, assess_titles):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

    return reduce_train, reduce_test, features
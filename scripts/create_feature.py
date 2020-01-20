from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from tqdm import tqdm
from collections import Counter
import json


def encode_title(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # typeとwordlを結合したカテゴリ変数の追加
    train['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['world']))
    test['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['world']))
    all_type_world = list(set(train["type_world"].unique()).union(test["type_world"].unique()))

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
    train['hour'] = train['timestamp'].dt.hour
    test['hour'] = test['timestamp'].dt.hour
    train['weekday'] = train['timestamp'].dt.weekday
    test['weekday'] = test['timestamp'].dt.weekday
    train['weekday_name'] = train['timestamp'].dt.weekday_name
    test['weekday_name'] = test['timestamp'].dt.weekday_name

    return train, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world


# Gameのmiss回数カウント
def cnt_miss(df):
    cnt = 0
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        y = json.loads(x)['misses']
        cnt += y
    return cnt


# event_code = 4020のaccuracy計算
def get_4020_acc(df, counter_dict, session_title_text):
    asses_title = ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)', 'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)']
    for e in asses_title:
        Assess_4020 = df[(df.event_code == 4020) & (session_title_text == e)]
        if len(Assess_4020) != 0:
            true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
            false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()
            measure_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (true_attempts_ + false_attempts_) != 0 else 0
            counter_dict[e + "_4020_accuracy"] += (counter_dict[e + "_4020_accuracy"] + measure_assess_accuracy_) / 2.0
    return counter_dict


def get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
            activities_labels, assess_titles, list_of_event_id, all_title_event_code,
            all_type_world, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    game_time_dict = {'Clip_gametime': 0, 'Game_gametime': 0, 'Activity_gametime': 0, 'Assessment_gametime': 0}
    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy': 0,
                            'Mushroom Sorter (Assessment)_4020_accuracy': 0,
                            'Bird Measurer (Assessment)_4020_accuracy': 0,
                            'Chest Sorter (Assessment)_4020_accuracy': 0}
    clip_comp_counts = {
                'Welcome to Lost Lagoon!':0,
                'Tree Top City - Level 1':0,
                'Ordering Spheres':0,
                'Costume Box':0,
                '12 Monkeys':0,
                'Tree Top City - Level 2':0,
                "Pirate's Tale":0,
                'Treasure Map':0,
                'Tree Top City - Level 3':0,
                'Rulers':0,
                'Magma Peak - Level 1':0,
                'Slop Problem':0,
                'Magma Peak - Level 2':0,
                'Crystal Caves - Level 1':0,
                'Balancing Act':0,
                'Lifting Heavy Things':0,
                'Crystal Caves - Level 2':0,
                'Honey Cake':0,
                'Crystal Caves - Level 3':0,
                'Heavy, Heavier, Heaviest':0
    }

    user_active_weekday = {'Sunday': 0, 'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0}
    user_active_hour = {'hour_'+str(i): 0 for i in range(25)}

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
    durations_clip = []
    durations_game = []
    durations_activity = []
    last_session_at = user_sample['timestamp'].min()  # 前のセッションの終了時刻

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    last_game_time_title = {'lgt_' + title: 0 for title in assess_titles}
    ac_game_time_title = {'agt_' + title: 0 for title in assess_titles}
    ac_true_attempts_title = {'ata_' + title: 0 for title in assess_titles}
    ac_false_attempts_title = {'afa_' + title: 0 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_code_proc_count = {str(ev) + "_proc" : 0. for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()}
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
    type_world_count: dict[str, int] = {w_eve: 0 for w_eve in all_type_world}

    # Game情報
    game_true_attempts = 0
    game_false_attempts = 0
    game_play_times = 0

    clip_time = 0

    Assessment_mean_event_count = 0
    Game_mean_event_count = 0
    Activity_mean_event_count = 0
    accumulated_game_miss = 0
    chest_assessment_uncorrect_sum = 0  # event_id == "df4fe8b6"の合計回数
    Cauldron_Filler_4025 = 0  # event_code == 4025 & title == "Cauldron Filler (Assessment)"のassessment accuracy
    session_count = 0

    # Gameイベントに含まれるそれぞれの特徴
    mean_game_round = 0
    max_game_round = 0
    cnt0_game_round = 0
    cnt1_game_round = 0
    cnt2_game_round = 0
    cnt3_game_round = 0
    cnt4_game_round = 0
    cnt5_game_round = 0
    cnt6_game_round = 0
    cnt7_game_round = 0
    cnt8_game_round = 0
    cnt9_game_round = 0
    cnt_over_game_round = 0
    mean_game_duration = 0
    max_game_duration = 0
    mean_game_level = 0
    max_game_level = 0
    cnt0_game_level = 0
    cnt1_game_level = 0
    cnt2_game_level = 0
    cnt3_game_level = 0
    cnt4_game_level = 0
    cnt5_game_level = 0
    cnt6_game_level = 0
    cnt7_game_level = 0
    cnt8_game_level = 0
    cnt9_game_level = 0
    cnt_over_game_level = 0
    mean_game_misses = 0
    max_game_misses = 0
    cnt0_game_misses = 0
    cnt1_game_misses = 0
    cnt2_game_misses = 0
    cnt3_game_misses = 0
    cnt4_game_misses = 0
    cnt5_game_misses = 0
    cnt6_game_misses = 0
    cnt7_game_misses = 0
    cnt8_game_misses = 0
    cnt9_game_misses = 0
    cnt_over_game_misses = 0

    user_sample['time_diff'] =  (user_sample['timestamp'].shift(periods=-1) - user_sample['timestamp']).dt.seconds # installation_idについて、次のtimestampとの差分

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]

        for weekday in user_active_weekday.keys():
            if weekday in session['weekday_name'].unique():
                    user_active_weekday[weekday] += 1

        for hour in user_active_hour.keys():
            if int(hour.replace('hour_', '')) in session['hour'].unique():
                user_active_hour[hour] += 1

        from_last_session = (last_session_at - session['timestamp'].min()).seconds/3600  # 前回セッションからの経過時間を計算
        last_session_at = session['timestamp'].max()  # 今のセッションの最後の時刻を記録（次のセッションに対しての計算時に使用）

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
            features.update(game_time_dict.copy())
            features.update(type_world_count.copy())
            features.update(event_code_proc_count.copy())
            features.update(last_game_time_title.copy())
            features.update(ac_game_time_title.copy())
            features.update(ac_true_attempts_title.copy())
            features.update(ac_false_attempts_title.copy())
            features.update(assess_4020_acc_dict.copy())
            features.update(clip_comp_counts.copy())

            features['installation_session_count'] = session_count
            features['hour'] = session['hour'].iloc[-1]
            features['weekday'] = session['weekday'].iloc[-1]
            features['from_last_session'] = from_last_session
            features['accumulated_game_miss'] = accumulated_game_miss

            features['mean_game_round'] = mean_game_round
            features['max_game_round'] = max_game_round
            features['cnt0_game_round'] = cnt0_game_round
            features['cnt1_game_round'] = cnt1_game_round
            features['cnt2_game_round'] = cnt2_game_round
            features['cnt3_game_round'] = cnt3_game_round
            features['cnt4_game_round'] = cnt4_game_round
            features['cnt5_game_round'] = cnt5_game_round
            features['cnt6_game_round'] = cnt6_game_round
            features['cnt7_game_round'] = cnt7_game_round
            features['cnt8_game_round'] = cnt8_game_round
            features['cnt9_game_round'] = cnt9_game_round
            features['cnt_over_game_round'] = cnt_over_game_round
            features['mean_game_duration'] = mean_game_duration
            features['max_game_duration'] = max_game_duration
            features['mean_game_level'] = mean_game_level
            features['max_game_level'] = max_game_level
            features['cnt0_game_level'] = cnt0_game_level
            features['cnt1_game_level'] = cnt1_game_level
            features['cnt2_game_level'] = cnt2_game_level
            features['cnt3_game_level'] = cnt3_game_level
            features['cnt4_game_level'] = cnt4_game_level
            features['cnt5_game_level'] = cnt5_game_level
            features['cnt6_game_level'] = cnt6_game_level
            features['cnt7_game_level'] = cnt7_game_level
            features['cnt8_game_level'] = cnt8_game_level
            features['cnt9_game_level'] = cnt9_game_level
            features['cnt_over_game_level'] = cnt_over_game_level
            features['mean_game_misses'] = mean_game_misses
            features['max_game_misses'] = max_game_misses
            features['cnt0_game_misses'] = cnt0_game_misses
            features['cnt1_game_misses'] = cnt1_game_misses
            features['cnt2_game_misses'] = cnt2_game_misses
            features['cnt3_game_misses'] = cnt3_game_misses
            features['cnt4_game_misses'] = cnt4_game_misses
            features['cnt5_game_misses'] = cnt5_game_misses
            features['cnt6_game_misses'] = cnt6_game_misses
            features['cnt7_game_misses'] = cnt7_game_misses
            features['cnt8_game_misses'] = cnt8_game_misses
            features['cnt9_game_misses'] = cnt9_game_misses
            features['cnt_over_game_misses'] = cnt_over_game_misses

            features['Assessment_mean_event_count'] = Assessment_mean_event_count
            features['Game_mean_event_count'] = Game_mean_event_count
            features['Activity_mean_event_count'] = Activity_mean_event_count
            features['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum

            # 各値のユニーク出現回数
            variety_features = [('var_event_code', event_code_count),
                                ('var_event_id', event_id_count),
                                ('var_title', title_count),
                                ('var_title_event_code', title_event_code_count),
                                ('var_type_world', type_world_count)]
            for name, dict_counts in variety_features:
                arr = np.array(list(dict_counts.values()))
                features[name] = np.count_nonzero(arr)

            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['game_session'] = i

            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts

            # 時間帯、曜日のユニークセッション数をカウント
            features.update(user_active_weekday.copy())
            features.update(user_active_hour.copy())

            # title毎のattempt
            ac_true_attempts_title['ata_' + session_title_text] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text] += false_attempts

            # 最後のAssessmentの時間と、Assessmentに要した総時間
            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]

            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
                features['duration_std'] = 0
                features['last_duration'] = 0
                features['duration_max'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
                features['last_duration'] = durations[-1]
                features['duration_max'] = np.max(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            if durations_clip == []:
                features['duration_clip_mean'] = 0
                features['duration_clip_std'] = 0
                features['clip_last_duration'] = 0
                features['clip_max_duration'] = 0
            else:
                features['duration_clip_mean'] = np.mean(durations_clip)
                features['duration_clip_std'] = np.std(durations_clip)
                features['clip_last_duration'] = durations_clip[-1]
                features['clip_max_duration'] = np.max(durations_clip)

            if durations_game == []:
                features['duration_game_mean'] = 0
                features['duration_game_std'] = 0
                features['game_last_duration'] = 0
                features['game_max_duration'] = 0
            else:
                features['duration_game_mean'] = np.mean(durations_game)
                features['duration_game_std'] = np.std(durations_game)
                features['game_last_duration'] = durations_game[-1]
                features['game_max_duration'] = np.max(durations_game)

            if durations_activity == []:
                features['duration_activity_mean'] = 0
                features['duration_activity_std'] = 0
                features['game_activity_duration'] = 0
                features['game_activity_max'] = 0
            else:
                features['duration_activity_mean'] = np.mean(durations_activity)
                features['duration_activity_std'] = np.std(durations_activity)
                features['game_activity_duration'] = durations_activity[-1]
                features['game_activity_max'] = np.max(durations_activity)

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

            # Gameの情報を追加
            features['game_num_incorrect'] = game_false_attempts
            features['game_num_correct'] = game_true_attempts
            features['game_play_again'] = game_play_times

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1]) / 2.0
            chest_assessment_uncorrect_sum += len(session[session.event_id == "df4fe8b6"])

            # Cauldron Filler (Assessment) かつ event_code == 4025のassess_accuracy
            features['Cauldron_Filler_4025'] = Cauldron_Filler_4025 / counter if counter > 0 else 0
            Assess_4025 = session[(session.event_code == 4025) & (session.title == 'Cauldron Filler (Assessment)')]
            true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
            false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()
            cau_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (true_attempts_ + false_attempts_) != 0 else 0
            Cauldron_Filler_4025 += cau_assess_accuracy_

            features['game_clip_time'] = session['game_time'].iloc[-1]/1000 + clip_time

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # Gameの情報を抽出
        if session_type == 'Game':
            game_true_attempts = session['info'].str.contains('Correct').sum()
            game_false_attempts = session['info'].str.contains('Incorrect').sum()
            game_play_times = session['info'].str.contains('again').sum()
            durations_game.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1]) / 2.0

            game_s = session[session.event_code == 2030]
            misses_cnt = cnt_miss(game_s)
            accumulated_game_miss += misses_cnt

            try:
                game_round = json.loads(session['event_data'].iloc[-1])["round"]
                mean_game_round = (mean_game_round + game_round) / 2.0
                max_game_round = max_game_round if max_game_round > game_round else game_round

                cnt0_game_round = cnt0_game_round + 1 if game_round == 0 else cnt0_game_round
                cnt1_game_round = cnt1_game_round + 1 if game_round == 1 else cnt1_game_round
                cnt2_game_round = cnt2_game_round + 1 if game_round == 2 else cnt2_game_round
                cnt3_game_round = cnt3_game_round + 1 if game_round == 3 else cnt3_game_round
                cnt4_game_round = cnt4_game_round + 1 if game_round == 4 else cnt4_game_round
                cnt5_game_round = cnt5_game_round + 1 if game_round == 5 else cnt5_game_round
                cnt6_game_round = cnt6_game_round + 1 if game_round == 6 else cnt6_game_round
                cnt7_game_round = cnt7_game_round + 1 if game_round == 7 else cnt7_game_round
                cnt8_game_round = cnt8_game_round + 1 if game_round == 8 else cnt8_game_round
                cnt9_game_round = cnt9_game_round + 1 if game_round == 9 else cnt9_game_round
                cnt_over_game_round = cnt_over_game_round + 1 if game_round > 9 else cnt_over_game_round
            except:
                pass

            try:
                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]
                mean_game_duration = (mean_game_duration + game_duration) / 2.0
                max_game_duration = max_game_duration if max_game_duration > game_duration else game_duration
            except:
                pass

            try:
                game_level = json.loads(session['event_data'].iloc[-1])["level"]
                mean_game_level = (mean_game_level + game_level) / 2.0
                max_game_level = max_game_level if max_game_level > game_level else game_level

                cnt0_game_level = cnt0_game_level + 1 if game_level == 0 else cnt0_game_level
                cnt1_game_level = cnt1_game_level + 1 if game_level == 1 else cnt1_game_level
                cnt2_game_level = cnt2_game_level + 1 if game_level == 2 else cnt2_game_level
                cnt3_game_level = cnt3_game_level + 1 if game_level == 3 else cnt3_game_level
                cnt4_game_level = cnt4_game_level + 1 if game_level == 4 else cnt4_game_level
                cnt5_game_level = cnt5_game_level + 1 if game_level == 5 else cnt5_game_level
                cnt6_game_level = cnt6_game_level + 1 if game_level == 6 else cnt6_game_level
                cnt7_game_level = cnt7_game_level + 1 if game_level == 7 else cnt7_game_level
                cnt8_game_level = cnt8_game_level + 1 if game_level == 8 else cnt8_game_level
                cnt9_game_level = cnt9_game_level + 1 if game_level == 9 else cnt9_game_level
                cnt_over_game_level = cnt_over_game_level + 1 if game_level > 9 else cnt_over_game_level
            except:
                pass

            try:
                game_misses = json.loads(session['event_data'].iloc[-1])["misses"]
                mean_game_misses = (mean_game_misses + game_misses) / 2.0
                max_game_misses = max_game_misses if max_game_misses > game_misses else game_misses

                cnt0_game_misses = cnt0_game_misses + 1 if game_misses == 0 else cnt0_game_misses
                cnt1_game_misses = cnt1_game_misses + 1 if game_misses == 1 else cnt1_game_misses
                cnt2_game_misses = cnt2_game_misses + 1 if game_misses == 2 else cnt2_game_misses
                cnt3_game_misses = cnt3_game_misses + 1 if game_misses == 3 else cnt3_game_misses
                cnt4_game_misses = cnt4_game_misses + 1 if game_misses == 4 else cnt4_game_misses
                cnt5_game_misses = cnt5_game_misses + 1 if game_misses == 5 else cnt5_game_misses
                cnt6_game_misses = cnt6_game_misses + 1 if game_misses == 6 else cnt6_game_misses
                cnt7_game_misses = cnt7_game_misses + 1 if game_misses == 7 else cnt7_game_misses
                cnt8_game_misses = cnt8_game_misses + 1 if game_misses == 8 else cnt8_game_misses
                cnt9_game_misses = cnt9_game_misses + 1 if game_misses == 9 else cnt9_game_misses
                cnt_over_game_misses = cnt_over_game_misses + 1 if game_misses > 9 else cnt_over_game_misses
            except:
                pass

        # Activityの情報を抽出
        if session_type == 'Activity':
            durations_activity.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1]) / 2.0

        # Clipの情報を抽出
        if session_type == 'Clip':
            clip_lengh = {
                'Welcome to Lost Lagoon!':19,
                'Tree Top City - Level 1':17,
                'Ordering Spheres':61,
                'Costume Box':61,
                '12 Monkeys':109,
                'Tree Top City - Level 2':25,
                "Pirate's Tale":80,
                'Treasure Map':156,
                'Tree Top City - Level 3':26,
                'Rulers':126,
                'Magma Peak - Level 1':20,
                'Slop Problem':60,
                'Magma Peak - Level 2':22,
                'Crystal Caves - Level 1':18,
                'Balancing Act':72,
                'Lifting Heavy Things':118,
                'Crystal Caves - Level 2':24,
                'Honey Cake':142,
                'Crystal Caves - Level 3':19,
                'Heavy, Heavier, Heaviest':61
            }
            clip_time += clip_lengh[session_title_text]

            durations_clip.append((clip_lengh[activities_labels[session_title]]))

            if session['time_diff'].iloc[0] >= clip_lengh[session_title_text]:
                clip_comp_counts[session_title_text] += 1  # Clipを最後まで見切ったかどうか

        session_count += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                x = k
                if col == 'title':
                    x = activities_labels[k]
                counter[x] += num_of_session_count[k]
            return counter

        def update_proc(count: dict):
            res = {}
            for k, val in count.items():
                res[str(k) + "_proc"] = (float(val) * 100.0) / accumulated_actions
            return res

        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')
        type_world_count = update_counters(type_world_count, 'type_world')
        assess_4020_acc_dict = get_4020_acc(session, assess_4020_acc_dict, session_title_text)
        game_time_dict[session_type + '_gametime'] = (game_time_dict[session_type + '_gametime'] + (session['game_time'].iloc[-1] / 1000.0)) / 2.0

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        event_code_proc_count = update_proc(event_code_count)

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
                        activities_labels, assess_titles, list_of_event_id,
                        all_title_event_code, all_type_world, is_kernel):
    compiled_train = []
    compiled_test = []
    if is_kernel:
        for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort = False)):
            # user_sampleに1つのins_idの全てのデータが入っている
            compiled_train += get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                    activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world)
        for ins_id, user_sample in test.groupby('installation_id', sort = False):
            test_data = get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                    activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world, test_set = True)
            compiled_test.append(test_data)
    else:
        for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
            # user_sampleに1つのins_idの全てのデータが入っている
            compiled_train += get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                    activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world)
        for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
            test_data = get_data(user_sample, win_code, list_of_user_activities, list_of_event_code,
                                    activities_labels, assess_titles, list_of_event_id, all_title_event_code, all_type_world, test_set = True)
            compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    # del reduce_train['weekday_name'], reduce_train['weekday_name']
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

        # Game情報のagg
        # df['installation_game_num_incorrect_mean'] = df.groupby(['installation_id'])['game_num_incorrect'].transform('mean')
        # df['installation_game_num_correct_mean'] = df.groupby(['installation_id'])['game_num_correct'].transform('mean')
        # df['installation_game_play_again_mean'] = df.groupby(['installation_id'])['game_play_again'].transform('mean')

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    reduce_train = reduce_train[features]
    reduce_test = reduce_test[features]
    # features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

    return reduce_train, reduce_test, features


def create_user_profile_train(df: pd.DataFrame) -> pd.DataFrame:
    # installation_id, session_order毎に、使用のあった日付のマスターを作成
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('H')
    df['train_labels_id'] = df['installation_id'] + '_' + df['session_order'].astype(str)
    user_active_date = df[['train_labels_id', 'timestamp']].drop_duplicates().reset_index(drop=True)
    user_active_date['hour'] = user_active_date['timestamp'].dt.hour
    user_active_date['weekday'] = user_active_date['timestamp'].dt.weekday_name
    user_active_date['date'] = user_active_date['timestamp'].dt.date

    # installation_id, session_order毎に、使用のあったhourの分布を抽出（日付のユニークカウントベース）
    user_active_hour = (
        user_active_date.groupby(['train_labels_id', 'hour'])['date']
            .nunique()  # ユニークな日付数をカウント
            .reset_index()
            .pivot(index='train_labels_id', columns='hour', values='date')
            .fillna(0)
    )
    user_active_hour.columns = ['user_active_at_' + str(col) for col in user_active_hour.columns]
    user_active_hour['installation_id'] = user_active_hour.index.map(lambda x: x.split('_')[0])
    user_active_hour = user_active_hour.groupby('installation_id').cumsum()  # installation_id毎に累積
    user_active_hour = user_active_hour + 5  # スムージング
    user_active_hour = user_active_hour / user_active_hour.sum(axis=0)  # 行ごとに合計が１となるように正規化

    # installation_id, session_orderごとに、使用のあった曜日の分布を抽出（日付のユニークカウントベース）
    user_active_weekday = (
        user_active_date.groupby(['train_labels_id', 'weekday'])['date']
            .nunique()  # ユニークな日付数をカウント
            .reset_index()
            .pivot(index='train_labels_id', columns='weekday', values='date')
            .fillna(0)
    )
    user_active_weekday.columns = ['user_active_at_' + str(col) for col in user_active_weekday.columns]
    user_active_weekday['installation_id'] = user_active_weekday.index.map(lambda x: x.split('_')[0])
    user_active_weekday = user_active_weekday.groupby('installation_id').cumsum()  # installation_id毎に累積
    user_active_weekday = user_active_weekday + 5  # スムージング
    user_active_weekday = user_active_weekday / user_active_weekday.sum(axis=0)  # 行ごとに合計が１となるように正規化

    # installation_id, session_orderごとに、アクティビティタイプの分布を抽出（セッションカウントベース）
    user_activity_type = pd.pivot_table(
        df.groupby(['train_labels_id', 'type'])['game_session'].nunique().reset_index(),
        values=['game_session'],
        index=['train_labels_id'],
        columns=['type'],
        fill_value=0
    )
    user_activity_type.columns = ['user_active_with_'+col for col in ['Activity', 'Assessment', 'Clip', 'Game']]
    user_activity_type['installation_id'] = user_activity_type.index.map(lambda x: x.split('_')[0])
    user_activity_type = user_activity_type.groupby('installation_id').cumsum()  # installation_id毎に累積
    user_activity_type = user_activity_type + 5  # スムージング
    user_activity_type = user_activity_type / user_activity_type.sum(axis=0)  # 行ごとに合計が１となるように正規化

    # installation_id, session_orderごとに、遊んだWorldの分布を抽出（セッションカウントベース）
    user_activity_world = pd.pivot_table(
        df.groupby(['train_labels_id', 'world'])['game_session'].nunique().reset_index(),
        values=['game_session'],
        index=['train_labels_id'],
        columns=['world'],
        fill_value=0
    )
    user_activity_world.columns = ['user_active_with_'+col for col in ['CRYSTALCAVES', 'MAGMAPEAK', 'NONE', 'TREETOPCITY']]
    user_activity_world['installation_id'] = user_activity_world.index.map(lambda x: x.split('_')[0])
    user_activity_world = user_activity_world.groupby('installation_id').cumsum()  # installation_id毎に累積
    user_activity_world = user_activity_world + 5  # スムージング
    user_activity_world = user_activity_world / user_activity_world.sum(axis=0)  # 行ごとに合計が１となるように正規化

    # 連結し、user属性として保持
    user_profile = (
        user_active_hour
            .merge(user_active_weekday, right_index=True, left_index=True)
            .merge(user_activity_type, right_index=True, left_index=True)
            .merge(user_activity_world, right_index=True, left_index=True)
    )

    # train_labelsへの結合用の列を抽出
    user_profile['installation_id'] = user_profile.index.map(lambda x: x.split('_')[0])
    user_profile['session_order'] = user_profile.index.map(lambda x: float(x.split('_')[1]))
    return user_profile


def create_user_profile_test(df: pd.DataFrame) -> pd.DataFrame:

    # installation_idごとに、使用のあった日付のマスターを作成
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('H')
    user_active_date = df[['installation_id', 'timestamp']].drop_duplicates().reset_index(drop=True)
    user_active_date['hour'] = user_active_date['timestamp'].dt.hour
    user_active_date['weekday'] = user_active_date['timestamp'].dt.weekday_name
    user_active_date['date'] = user_active_date['timestamp'].dt.date

    # installation_idごとに、使用のあったhourの分布を抽出（日付のユニークカウントベース）
    user_active_hour = (
        user_active_date.groupby(['installation_id', 'hour'])['date']
                        .nunique()  # ユニークな日付数をカウント
                        .reset_index()
                        .pivot(index='installation_id', columns='hour', values='date')
                        .fillna(0)
    )
    user_active_hour.columns = ['user_active_at_' + str(col) for col in user_active_hour.columns]
    user_active_hour = user_active_hour + 5  # スムージング
    user_active_hour = user_active_hour / user_active_hour.sum(axis=0)  # installation_idごとに合計が１となるように正規化

    # installation_idごとに、使用のあった曜日の分布を抽出（日付のユニークカウントベース）
    user_active_weekday = (
        user_active_date.groupby(['installation_id', 'weekday'])['date']
                        .nunique()  # ユニークな日付数をカウント
                        .reset_index()
                        .pivot(index='installation_id', columns='weekday', values='date')
                        .fillna(0)
    )
    user_active_weekday.columns = ['user_active_at_' + str(col) for col in user_active_weekday.columns]
    user_active_weekday = user_active_weekday + 5  # スムージングuser_active_hour = user_active_hour + 5  # スムージング
    user_active_weekday = user_active_weekday / user_active_weekday.sum(axis=0)  # installation_idごとに合計が１となるように正規化

    # installation_idごとにアクティビティタイプの分布を抽出（セッションカウントベース）
    user_activity_type = pd.pivot_table(
        df.groupby(['installation_id', 'type'])['game_session'].nunique().reset_index(),
        values=['game_session'],
        index=['installation_id'],
        columns=['type'],
        fill_value=0
    )
    user_activity_type.columns = ['user_active_with_' + col for col in ['Activity', 'Assessment', 'Clip', 'Game']]
    user_activity_type = user_activity_type.groupby('installation_id').cumsum()  # installation_id毎に累積
    user_activity_type = user_activity_type + 5  # スムージング
    user_activity_type = user_activity_type / user_activity_type.sum(axis=0)  # 行ごとに合計が１となるように正規化

    # installation_id, session_orderごとに、遊んだWorldの分布を抽出（セッションカウントベース）
    user_activity_world = pd.pivot_table(
        df.groupby(['installation_id', 'world'])['game_session'].nunique().reset_index(),
        values=['game_session'],
        index=['installation_id'],
        columns=['world'],
        fill_value=0
    )
    user_activity_world.columns = ['user_active_with_'+col for col in ['CRYSTALCAVES', 'MAGMAPEAK', 'NONE', 'TREETOPCITY']]
    user_activity_world = user_activity_world + 5  # スムージング
    user_activity_world = user_activity_world / user_activity_world.sum(axis=0)  # 行ごとに合計が１となるように正規化

    # 連結し、user属性として保持
    user_profile = (
        user_active_hour
            .merge(user_active_weekday, right_index=True, left_index=True)
            .merge(user_activity_type, right_index=True, left_index=True)
            .merge(user_activity_world, right_index=True, left_index=True)
    ).reset_index()  # installation_idをカラムとして保持
    return user_profile


def add_session_order_to_train(train, train_labels):
    """
    trainのログをどのtrain_labelsにあるアセスメントセッションに影響を与えられるのか明示し、リークを防ぐためにsession_orderを定義する。
    その後、train_labelsのセッション単位で集計をかけたい。
    """
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    game_session_start_time = train.groupby(['game_session'])['timestamp'].min().reset_index()  # セッション毎の開始時刻を集計

    # train_labelsにあるアセスメントセッションの開始時刻とそれを元にしたorderを付与
    train_session_master = train_labels.merge(game_session_start_time, how='left', on=['game_session']).sort_values(
        by=['installation_id', 'timestamp'])
    train_session_master['session_order'] = 1
    train_session_master['session_order'] = train_session_master.groupby(['installation_id'])['session_order'].cumsum()

    # trainにそれぞれの行動がどのsession_orderに属するか付与
    train = train.merge(train_session_master[['game_session', 'session_order']], on=['game_session'], how='left')
    train['session_order'] = (train
                              .sort_values(by=['installation_id', 'timestamp'])
                              .groupby('installation_id')['session_order']
                              .fillna(method='bfill')  # installation_id毎に、特定のアセスメントの前の行動は同じsession_orderを当てはめる
                              )
    train.dropna(subset=['session_order'], inplace=True)
    train_session_master = train_session_master[['installation_id', 'game_session', 'session_order']]
    return train, train_session_master

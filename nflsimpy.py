""" nflsimpy.py: This program builds a classifier that predicts a football play
                      call given various input parameters.  The program then generates
                      outcome probabilities for different play calls for given input
                      teams.  Using the classifier and outcome probabilities, football
                      drives are simulated. """

#import math
import random
import warnings
#import graphviz
import os.path
import numpy as np
import pandas as pd
#import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import gamma
#from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from copy import deepcopy
#from scipy.stats import norm
#from sklearn.tree import export_graphviz


def reimport_data(reimport = False):
    """ Football play-by-play data is imported into a pandas dataframe """

    if os.path.isfile('data/coach.csv') and reimport == False:
        coach_df = pd.read_csv('data/coach.csv')
    else:
        nfl_df = pd.read_csv('data/NFL by Play 2009-2016 (v2).csv')
        coach_df = nfl_df[['PlayType', 'GameID', '\ufeffDate',
                           'Drive', 'qtr', 'down', 'time', 'PlayTimeDiff',
                           'yrdln', 'yrdline100','ydstogo',
                           'posteam', 'DefensiveTeam', 'PosTeamScore', 'DefTeamScore',
                           'PassLength', 'PassLocation', 'PassAttempt','AirYards', 'PassOutcome',
                           'RushAttempt', 'RunLocation', 'RunGap', 'Yards.Gained',
                           'Sack','Fumble','InterceptionThrown', 'RecFumbTeam',
                           'FieldGoalDistance','FieldGoalResult']].copy()
        
        ###########################################
        # Generate data for elapsed time per play #
        ###########################################
        game_id = coach_df['GameID']
        elapsed_time = coach_df['PlayTimeDiff']
        elapsed_play_time = np.empty((len(elapsed_time)))
        elapsed_play_time[:] = np.nan
        for i,game in enumerate(game_id[:-1]):
            if game_id[i+1] == game:
                elapsed_play_time[i] = elapsed_time[i+1]
        coach_df['Elapsed_Play_Time'] = elapsed_play_time

        ##############################################
        # Generate data for return spot after a punt #
        ##############################################
        play_type = coach_df['PlayType']
        yardline = coach_df['yrdline100']
        return_spot = np.empty((len(play_type)))
        return_spot[:] = np.nan
        for i,play in enumerate(play_type):
            if play == 'Punt':
                return_spot[i] = yardline[i+1]
        coach_df['Return_spot'] = return_spot

        #########################
        # Save dataframe to csv #
        #########################
        coach_df.to_csv('data/coach.csv')

    return coach_df

class Team:

    def __init__(self, team_name, play_by_play_df):
        self.team = team_name
        self.team_df = play_by_play_df[play_by_play_df['posteam'] == self.team]
        self._generate_lists()

        self.valid_play_dict = {'Pass': 0, 'Run': 1, 'Punt': 2, 'Field Goal': 3}
        self.valid_play_inv_dict = {0: 'Pass', 1: 'Run', 2: 'Punt', 3: 'Field Goal'}

        self.X = []
        self.Y = []

    def train_classifier(self, debug_classifier = False):
        self._organize_training_data()
        self._generate_random_forest(debug_classifier)

    def _generate_random_forest(self, debug_classifier):
        self.forest = RandomForestClassifier(n_estimators=100, random_state=1)
        self.multi_target_forest = MultiOutputClassifier(self.forest, n_jobs=-1)
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=0)
        self.multi_target_forest.fit(X_train, Y_train)

        forests = self.multi_target_forest.estimators_
        forest0_feat = forests[0].feature_importances_.tolist()
        forest1_feat = forests[1].feature_importances_.tolist()
        forest2_feat = forests[2].feature_importances_.tolist()
        forest3_feat = forests[3].feature_importances_.tolist()

        feature_df = pd.DataFrame(data = {'Features': [x for x in range(5)],
                                          'Forest0': forest0_feat,
                                          'Forest1': forest1_feat,
                                          'Forest2': forest2_feat,
                                          'Forest3': forest3_feat})

        if debug_classifier == True:
            print('Training Score: ', self.multi_target_forest.score(X_train, Y_train))
            print('Test Score: ', self.multi_target_forest.score(X_test, Y_test))

            fig1 = plt.figure()

            ax = fig1.add_subplot(111) 

            width = 0.1

            feature_df.Forest0.plot(kind='bar', color='red', ax=ax, width=width, position=-1)
            feature_df.Forest1.plot(kind='bar', color='green', ax=ax, width=width, position=0)
            feature_df.Forest2.plot(kind='bar', color='blue', ax=ax, width=width, position=1)
            feature_df.Forest3.plot(kind='bar', color='yellow', ax=ax, width=width, position=2)

            ax.set_xticklabels(['Yards to First', 'Down', 'Quarter', 'Yardline','Score Diff'], rotation = 0)
            ax.set_xlabel('Features')
            ax.set_ylabel('Feature Importance')
            ax.set_title('Random Forest - Feature Analysis')

            plt.xlim(-0.5, 4.5)
            plt.legend(['Pass', 'Run', 'Punt', 'Field Goal'])
            plt.show()

    def test_classifier(self,yards_to_go, down, quarter, yard_line, score_diff):

        input_array = np.array([yards_to_go, down, quarter, yard_line, score_diff])
        prediction = self.multi_target_forest.predict_proba(input_array).tolist()
        prediction = prediction[0][1]
        return np.argmax(prediction)

    def _generate_lists(self):

        self.play_type = self.team_df['PlayType'].values.tolist()
        self.game_ID = self.team_df['GameID'].values.tolist()
        self.drive = self.team_df['Drive'].values.tolist()
        self.quarter = self.team_df['qtr'].values.tolist()
        self.down = self.team_df['down'].values.tolist()
        self.time = self.team_df['time'].values.tolist()
        self.pos_team = self.team_df['posteam'].values.tolist()
        self.def_team = self.team_df['DefensiveTeam'].values.tolist()
        self.pass_length = self.team_df['PassLength'].values.tolist()
        self.pass_location = self.team_df['PassLocation'].values.tolist()
        self.pass_attempt = self.team_df['PassAttempt'].values.tolist()
        self.air_yards = self.team_df['AirYards'].values.tolist()
        self.rush_attempt = self.team_df['RushAttempt'].values.tolist()
        self.run_location = self.team_df['RunLocation'].values.tolist()
        self.run_gap = self.team_df['RunGap'].values.tolist()
        self.fieldgoal_distance = self.team_df['FieldGoalDistance'].values.tolist()
        self.pos_team_score = self.team_df['PosTeamScore'].values.tolist()
        self.def_team_score = self.team_df['DefTeamScore'].values.tolist()
        self.yrdline100 = self.team_df['yrdline100'].values.tolist()
        self.yrds_to_go = self.team_df['ydstogo'].values.tolist()


    def _organize_training_data(self):

        score_diff_list = np.array(self.pos_team_score) - np.array(self.def_team_score)
        zipped_data = zip(self.quarter, self.down, self.yrdline100, self.yrds_to_go, score_diff_list, self.play_type)

        for quarter,down,yrdln,yrds_to_go, score_diff, play_type in zipped_data:

            input_list = [yrds_to_go, down, quarter, yrdln, score_diff]
            if not np.any(np.isnan(input_list)) and play_type in self.valid_play_dict:

                output_list = [0 for _ in range(4)]
                output_list[self.valid_play_dict[play_type]] = 1

                self.X.append(input_list)
                self.Y.append(output_list)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def generate_success_probabilities(self, opponent, yr, debug_probs = False):
        ##############################
        # Extract Team Specific Data #
        ##############################
        self.opponent = opponent

        valid_dates = [str(yr) + '-' + '09',
                       str(yr) + '-' + '10',
                       str(yr) + '-' + '11',
                       str(yr) + '-' + '12',
                       str(yr + 1) + '-' + '01']

        coach_yr_09_df = self.team_df[self.team_df['\ufeffDate'].str.contains(valid_dates[0])]
        coach_yr_10_df = self.team_df[self.team_df['\ufeffDate'].str.contains(valid_dates[1])]
        coach_yr_11_df = self.team_df[self.team_df['\ufeffDate'].str.contains(valid_dates[2])]
        coach_yr_12_df = self.team_df[self.team_df['\ufeffDate'].str.contains(valid_dates[3])]
        coach_yr_01_df = self.team_df[self.team_df['\ufeffDate'].str.contains(valid_dates[4])]

        coach_yr_df = pd.concat([coach_yr_09_df, coach_yr_10_df, coach_yr_11_df, coach_yr_12_df, coach_yr_01_df])

        team_prob_df = coach_yr_df[coach_yr_df['DefensiveTeam'] == self.opponent]

        loc_pass_outcome = team_prob_df['PassOutcome'].values.tolist()
        loc_yrds_gained = team_prob_df['Yards.Gained'].values.tolist()
        loc_play_type = team_prob_df['PlayType'].values.tolist()
        loc_interception = team_prob_df['InterceptionThrown'].values.tolist()

        loc_play_type_fumble = coach_yr_df['PlayType'].values.tolist()
        loc_fumble = coach_yr_df['Fumble'].values.tolist()
        loc_drive = coach_yr_df['Drive'].values.tolist()
        loc_gameID = coach_yr_df['GameID'].values.tolist()

        loc_fg_success = coach_yr_df['FieldGoalResult']
        loc_fg_distance = coach_yr_df['yrdline100']
        loc_fg_play_type = coach_yr_df['PlayType']

        loc_punt_spot = coach_yr_df['yrdline100']
        loc_punt_return = coach_yr_df['Return_spot']

        loc_time_elapsed = coach_yr_df['Elapsed_Play_Time']

        ########################
        # Initialize Variables #
        ########################
        self.elapsed_time = {'punt': [], 'run': [], 'pass_good': [], 'pass_nogood': [], 'fg': []}

        self.total_passes = 0
        self.total_completions = 0
        self.pass_list = []
        self.rush_list = []

        self.pass_or_sack = 0
        self.num_sacks = 0
        self.sack_dist = []

        self.total_interceptions = 0

        field_goal_attempts = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0}
        field_goal_successes = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0}
        self.field_goal_pct = {}

        total_runs = 0
        total_run_fumbles = 0
        total_pass = 0
        total_pass_fumbles = 0

        self.punt_dist = []
        punt_touchback = {90:0, 80:0, 70:0, 60:0, 50:0, 40: 0, 30: 0, 20:0}
        punt_kickrange = {90:0, 80:0, 70:0, 60:0, 50:0, 40: 0, 30: 0, 20:0}
        punt_total = 0

        #####################
        # Punt Calculations #
        #####################
        for punt_spot, return_spot, time in zip(loc_punt_spot, loc_punt_return, loc_time_elapsed):
            if np.isnan(punt_spot) == False and np.isnan(return_spot) == False:
                punt_total +=1
                punt_range = np.floor(punt_spot / 10) * 10
                punt_kickrange[punt_range] +=1
                if return_spot == 80:
                    punt_touchback[punt_range] +=1
                else:
                    self.punt_dist.append(return_spot - (100-punt_spot))
                if np.isnan(time) == False:
                    self.elapsed_time['punt'].append(time)
        self.punt_alpha, self.punt_loc, self.punt_beta = stats.gamma.fit(self.punt_dist)
        punt_x = np.arange(-10, 80, 1)
        g3 = gamma.pdf(x=punt_x, a=self.punt_alpha, loc=self.punt_loc, scale=self.punt_beta)

        self.punt_touchback_pct = {}
        for key,value in punt_kickrange.items():
            if value != 0:
                self.punt_touchback_pct[key] = punt_touchback[key]/value

        ###########################
        # Field Goal Calculations #
        ###########################
        for fg_success, fg_distance, fg_play_type, time in zip(loc_fg_success, loc_fg_distance, loc_fg_play_type, loc_time_elapsed):

            if fg_play_type == 'Field Goal':
                marker = np.floor(fg_distance/10)*10
                if marker is not None:
                    if np.isnan(time) == False:
                        self.elapsed_time['fg'].append(time)
                    field_goal_attempts[marker] += 1
                    if fg_success == 'Good':
                        field_goal_successes[marker] += 1

        for key,value in field_goal_attempts.items():
            if value > 0:
                self.field_goal_pct[key] = field_goal_successes[key]/value
            else:
                self.field_goal_pct[key] = 0

        #######################
        # Fumble Calculations #
        #######################
        for i, fumble in enumerate(loc_fumble):
            current_game = loc_gameID[i]
            current_drive = loc_drive[i]
            if loc_play_type_fumble[i] == 'Pass':
                total_pass += 1
                if fumble == 1:
                    if loc_gameID[i+1] == current_game:
                        if loc_drive[i+1] == current_drive or loc_drive[i+1] == current_drive + 1:
                            pass
                        else:
                            total_pass_fumbles +=1
            elif loc_play_type_fumble[i] == 'Run':
                total_runs += 1
                if fumble == 1:
                    if loc_gameID[i+1] == current_game:
                        if loc_drive[i+1] == current_drive or loc_drive[i+1] == current_drive + 1:
                            pass
                        else:
                            total_run_fumbles +=1

        self.pass_fumble_pct = total_pass_fumbles/total_pass
        self.run_fumble_pct = total_run_fumbles/total_runs

        #############################
        # Pass and Run Calculations #
        #############################
        for pass_outcome, yrds_gained, play_type, interception, time in zip(loc_pass_outcome,
                                                                         loc_yrds_gained, loc_play_type,
                                                                         loc_interception, loc_time_elapsed):

            if play_type == 'Pass' or play_type == 'Sack':
                self.pass_or_sack += 1
                if play_type == 'Sack':
                    self.num_sacks += 1
                    self.sack_dist.append(yrds_gained)

            if play_type == 'Pass':
                self.total_passes += 1
                if pass_outcome == "Complete":
                    self.total_completions += 1
                    self.pass_list.append(yrds_gained)
                    if np.isnan(time) == False:
                        self.elapsed_time['pass_good'].append(time)
                else:
                    if np.isnan(time) == False:
                        self.elapsed_time['pass_nogood'].append(time)
                if interception == 1:
                    self.total_interceptions +=1

            elif play_type == 'Run':
                if np.isnan(time) == False:
                    self.elapsed_time['run'].append(time)
                self.rush_list.append(yrds_gained)

        self.time_kde = {}

        self.time_kde['pass_good'] = stats.gaussian_kde(self.elapsed_time['pass_good'], bw_method=.2)
        self.time_kde['pass_nogood'] = stats.gaussian_kde(self.elapsed_time['pass_nogood'], bw_method=.2)
        self.time_kde['punt'] = stats.gaussian_kde(self.elapsed_time['punt'], bw_method=.2)
        self.time_kde['run'] = stats.gaussian_kde(self.elapsed_time['run'], bw_method=.2)
        self.time_kde['fg'] = stats.gaussian_kde(self.elapsed_time['fg'], bw_method=.2)

        self.pass_complete_pct = self.total_completions / self.total_passes

        self.pass_alpha, self.pass_loc, self.pass_beta = stats.gamma.fit(self.pass_list)  
        self.run_alpha, self.run_loc, self.run_beta = stats.gamma.fit(self.rush_list)

        self.sack_pct = self.num_sacks / self.pass_or_sack
        self.sack_yrds_mean = np.mean(self.sack_dist)
        self.sack_yrds_std = np.std(self.sack_dist)
        self.interception_pct = self.total_interceptions/ self.total_passes

        #############
        # Debugging #
        #############
        if debug_probs == True:
            pass_x = np.arange(0,40,.1)
            g1 = gamma.pdf(x=pass_x, a=self.pass_alpha, loc=self.pass_loc, scale=self.pass_beta)

            run_x = np.arange(-10,20,.1)
            g2 = gamma.pdf(x=run_x,a=self.run_alpha,loc=self.run_loc,scale=self.run_beta)

            fig2 = plt.figure()

            ax1 = fig2.add_subplot(2,1,1)
            ax1.plot(pass_x, g1)
            ax1.hist(self.pass_list, bins=20, normed=True)
            ax1.set_xlabel('Pass Yards')
            ax1.set_ylabel('Probability')

            ax2 = fig2.add_subplot(2,1,2)
            ax2.plot(run_x, g2)
            ax2.hist(self.rush_list, 20, normed=True)
            ax2.set_xlabel('Rush Yards')
            ax2.set_ylabel('Probability')
            fig2.show()

            fig3 = plt.figure()

            ax3 = fig3.add_subplot(1,1,1)
            ax3.plot(punt_x,g3)
            ax3.hist(self.punt_dist,bins=20,normed=True)
            fig3.show()

            fig6 = plt.figure()

            ax6 = fig6.add_subplot(1,1,1)
            print('TIMES', self.elapsed_time)
            for key,value in self.elapsed_time.items():
                ax6.hist(value, histtype = 'step', label = key)
            ax6.legend()
            fig6.show()

def game_simulator(team1,team2, plot_sim = True, verbose = True):
    ###################
    # Initialize Game #
    ###################
    offense = deepcopy(team1)
    defense = deepcopy(team2)

    team_ind = 1

    num_tds = 0

    plot_x = []
    plot_y = []

    simulate_num = 1

    team_play_list = []
    play_result_list = []

    quarter = 0

    play_num = 1

    ########################
    # Loop Until Game Over #
    ########################
    while(quarter <= 3):
        quarter += 1
        quarter_time = 60 * 15

        while(quarter_time > 0):

            ####################
            # Initialize Drive #
            ####################
            if ((quarter == 2 or quarter == 4) and quarter_time == 60*15 and series_done == False):
                pass
            else:
                yardline = 80
                down = 1
                yards_to_go = 10
                score_diff = 0
                series_done = False
                drive_done = False

            ###########################
            # Loop Until Score Occurs #
            ###########################
            while (series_done == False and quarter_time > 0):

                yards_gained = 0

                team_play_list.append(team_ind)

                if team_ind == 1:
                    play_loc_x = [yardline]
                    play_loc_y = [play_num]
                else:
                    play_loc_x = [100 - yardline]
                    play_loc_y = [play_num]

                #############
                # Play Call #
                #############
                next_play = offense.test_classifier(yards_to_go, down, quarter, yardline, score_diff)

                if verbose == True:
                    print('Current Down: ', down,
                          '   Yards to go: ', yards_to_go,
                          '   Yardline: ', yardline,
                          '   Next Play: ', offense.valid_play_inv_dict[next_play],
                          '   Drive Val: ', play_num)

                ####################
                # Pass  Simulation #
                ####################
                if offense.valid_play_inv_dict[next_play] == "Pass":
                    sacked = np.random.uniform(0, 1, 1)
                    pass_success = np.random.uniform(0, 1, 1)
                    intercept_success = np.random.uniform(0, 1, 1)
                    if sacked <= offense.sack_pct:
                        time_elapsed = offense.time_kde['pass_nogood'].resample(1)[0][0]
                        yards_gained = random.gauss(offense.sack_yrds_mean, offense.sack_yrds_std)
                        yardline -= yards_gained
                        yards_to_go -= yards_gained
                        if yards_to_go <= 0:
                            yards_to_go = 10
                            down = 1
                        elif down < 4:
                            down += 1
                        else:
                            drive_done = True
                        play_result_list.append('Pass')
                    elif pass_success >= 1 - offense.pass_complete_pct:
                        fumble_occurred = np.random.uniform(0, 1, 1)
                        if fumble_occurred < offense.pass_fumble_pct:
                            time_elapsed = offense.time_kde['pass_nogood'].resample(1)[0][0]
                            drive_done = True
                            yards_gained = 0
                            play_result_list.append('Fumble')
                        else:
                            time_elapsed = offense.time_kde['pass_good'].resample(1)[0][0]
                            yards_gained = random.gammavariate(offense.pass_alpha, offense.pass_beta) + offense.pass_loc
                            yardline -= yards_gained
                            yards_to_go -= yards_gained
                            play_result = 'Pass'
                            if yardline <= 0:
                                yardline = 0
                                num_tds += 1
                                series_done = True
                                play_result = 'Touchdown'
                            elif yards_to_go <= 0:
                                yards_to_go = 10
                                down = 1
                            elif down < 4:
                                down += 1
                            else:
                                drive_done = True
                            play_result_list.append(play_result)
                    elif intercept_success >= 1 - offense.interception_pct:
                        time_elapsed = offense.time_kde['pass_nogood'].resample(1)[0][0]
                        drive_done = True
                        yards_gained = 0
                        play_result_list.append('Interception')
                    else:
                        time_elapsed = offense.time_kde['pass_nogood'].resample(1)[0][0]
                        yards_gained = 0
                        if down < 4:
                            down += 1
                        else:
                            drive_done = True
                        play_result_list.append('Pass')

                ##################
                # Run Simulation #
                ##################
                if offense.valid_play_inv_dict[next_play] == "Run":
                    fumble_occurred = np.random.uniform(0, 1, 1)
                    time_elapsed = offense.time_kde['run'].resample(1)[0][0]
                    if fumble_occurred < offense.run_fumble_pct:
                        drive_done = True
                        yards_gained = 0
                        play_result_list.append('Fumble')
                    else:
                        yards_gained = random.gammavariate(offense.run_alpha, offense.run_beta) + offense.run_loc
                        yardline -= yards_gained
                        yards_to_go -= yards_gained
                        play_result = 'Run'
                        if yardline <= 0:
                            yardline = 0
                            num_tds += 1
                            series_done = True
                            play_result = 'Touchdown'
                        elif yards_to_go <= 0:
                            yards_to_go = 10
                            down = 1
                        elif down < 4:
                            down += 1
                        else:
                            drive_done = True
                        play_result_list.append(play_result)

                ###################
                # Punt Simulation #
                ###################
                if offense.valid_play_inv_dict[next_play] == "Punt":
                    time_elapsed = offense.time_kde['punt'].resample(1)[0][0]
                    punt_touchback_random = np.random.uniform(0,1,1)
                    marker = np.floor(yardline/10) * 10
                    if punt_touchback_random < offense.punt_touchback_pct[marker]:
                        yardline = 20
                    else:
                        punt_yardline = random.gammavariate(offense.punt_alpha, offense.punt_beta) + offense.punt_loc
                        yardline -= punt_yardline
                        if yardline <= 0:
                            yardline = 20
                    drive_done = True
                    play_result_list.append('Punt')

                #########################
                # Field Goal Simulation #
                #########################
                if offense.valid_play_inv_dict[next_play] == "Field Goal":
                    time_elapsed = offense.time_kde['fg'].resample(1)[0][0]
                    field_goal_random = np.random.uniform(0, 1, 1)
                    marker = np.floor(yardline / 10) * 10
                    field_goal_prob = offense.field_goal_pct[marker]
                    if field_goal_random >= 1 - field_goal_prob:
                        series_done = True
                        yardline = 0
                        play_result_list.append('FG_Good')
                    else:
                        drive_done = True
                        play_result_list.append('FG_NoGood')

                

                ##########################
                # Field Direction Change #
                ##########################
                if drive_done == True or series_done == True:
                    drive_done = False
                    yardline = 100 - yardline
                    yards_to_go = 10
                    offense_temp = deepcopy(offense)
                    defense_temp = deepcopy(defense)
                    offense = deepcopy(defense_temp)
                    defense = deepcopy(offense_temp)
                    down = 1
                    if team_ind == 1:
                        team_ind = 2
                    else:
                        team_ind = 1

                ###################
                # Time Adjustment #
                ###################
                quarter_time -=  time_elapsed

                ###########
                # Display #
                ###########
                if verbose == True:
                    print('   Play Called: ', play_result_list[-1])
                    print('   Yards Gained: ', yards_gained)
                    print('   Time Remaining in Quarter: ', quarter_time)
                    print('   Quarter #: ', quarter)
                
                ##################
                # Plotting Setup #
                ##################
                if plot_sim == True:
                    if team_ind == 1:
                        play_loc_x.append(yardline)
                        play_loc_y.append(play_num)
                    else:
                        play_loc_x.append(100 - yardline)
                        play_loc_y.append(play_num)

                    plot_x.append(play_loc_x)
                    plot_y.append(play_loc_y)

                ###########################
                # Increment for Next Play #
                ###########################
                play_num += 1

   #############
    # Plotting #
    ############
    if plot_sim == True:

        fig3 = plt.figure()
        ax3= fig3.add_subplot(111)

        for px, py, team, result in zip(plot_x, plot_y, team_play_list, play_result_list):
            if team == 1:
                team_color = 'y'
            else:
                team_color = 'k'

            if result == 'Run' or result == 'Pass':
                ax3.plot(px, py, team_color)
                ax3.plot(px[-1], py[-1], 'ko', gid = result)
            elif result == 'Touchdown':
                ax3.plot(px, py, 'g')
            elif result == 'Fumble' or result == 'Interception':
                ax3.plot(px, py, team_color)
                ax3.plot(px[-1],py[-1],'rx', gid = result)
            elif result == 'Punt':
                ax3.plot(px, py, team_color)
                ax3.plot(px[-1], py[-1], 'ro', gid = result)
            elif result == 'FG_Good':
                ax3.plot(px, py, 'g--', gid = result)
            elif result == 'FG_NoGood':
                ax3.plot(px, py, 'cx', gid = result)

        dummy_td = ax3.plot([], [], 'g', label='Touchdown')
        dummy_fgng = ax3.plot([], [], 'cx', label='Field Goal No Good')
        dummy_fgg = ax3.plot([], [], 'g--', label='Field Goal Good')
        dummy_fumint = ax3.plot([], [], 'rx', label='Fumble/Interception')
        dummy_punt = ax3.plot([], [], 'ro', label='Punt')
        dummy_turnover = ax3.plot([], [], 'bo', label='Turnover')

        def on_plot_hover(event):
            for curve in ax3.get_lines():
                if curve.contains(event)[0]:
                    if curve.get_gid() is not None:
                        print(curve.get_gid())

        fig3.canvas.mpl_connect('motion_notify_event', on_plot_hover) 

        plt.gca().invert_xaxis()
        plt.grid()
        plt.legend()
        plt.ylabel('Play #')
        plt.xlabel('Distance to Endzone')
        plt.xlim([100, 0])
        plt.ylim(ymin=1)
        start, end = plt.gca().get_ylim()
        plt.gca().set_yticks(np.arange(start, end, 1))
        plt.title('2016 NY Giants Vs Dallas Cowboys')
        plt.show()

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    coach_df = reimport_data()

    NYG = Team('NYG',coach_df)
    NYG.train_classifier()
    NYG.generate_success_probabilities('DAL',2016)

    DAL = Team('DAL',coach_df)
    DAL.train_classifier()
    DAL.generate_success_probabilities('NYG',2016)

    game_simulator(NYG,DAL)

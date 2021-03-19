import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import music21
import tqdm
import seaborn as sns
from pprint import pprint
import seaborn as sns 
from scipy.signal import resample
from collections import Counter
from .turkish_makam import ScoreSet, Makam, Usul

def gen_avg_note_duration_vs_usul(alignment_dict: dict, plots_dir: str):
    eighth_notes_dur = []
    sixteenth_notes_dur = []
    all_notes_dur = []
    hits_dur = []

    for makam_usul_key in tqdm.tqdm(alignment_dict.keys()):
        df = alignment_dict[makam_usul_key]['song_to_melodic_outline_df']
        df = df[df.offset != 0.0]

        hits_dur += df.duration.to_list()
        usul_visitations_n = df.shape[0]
        time_signature_occ_count = Counter()

        for time_signature_candidate, time_signature_df in df.groupby(['time_sig']):
            time_signature_occ_count[time_signature_candidate] = time_signature_df.shape[0]
        
        if not usul_visitations_n:
            continue

        makam_usul_specific_time_signature = time_signature_occ_count.most_common()[0][0]
        numerator = makam_usul_specific_time_signature.split('/')[0]
        denominator = makam_usul_specific_time_signature.split('/')[1]
        usul_ = Usul(makam_usul_key[1], makam_usul_specific_time_signature)
        eighth_in_measure_n = int(numerator) * (2 if denominator == '4' else 1)
        bar_length = eighth_in_measure_n * 0.5
        
        # Counts all notes that happen on a sixteenth vs eighth grid
        eighth_notes_count = 0
        sixteenth_notes_count = 0

        for score in alignment_dict[makam_usul_key]['scores_dict'].values():
            score_makam_part = score.parts[0].flat.getElementsByClass(music21.note.Note)

            eighth_notes_dur += [note.duration.quarterLength for note in
                                score_makam_part if note.offset % 1 in [0, 0.5] and
                                note.offset % bar_length != 0.0]
            
            sixteenth_notes_dur += [note.duration.quarterLength for note in
                                    score_makam_part if note.offset % 1 in [0, 0.25, 0.5, 0.75] and note.offset % bar_length != 0.0]
            
            all_notes_dur += [note.duration.quarterLength for note in
                            score_makam_part if note.offset % bar_length != 0.0]
        
    avg_eighth_notes_dur = round(np.mean(eighth_notes_dur), 2)
    avg_sixteenth_notes_dur = round(np.mean(sixteenth_notes_dur), 2)
    avg_hits_dur = round(np.mean(hits_dur), 2)
    hits_n = len(hits_dur)
    eighth_notes_n = len(eighth_notes_dur)
    sixteenth_notes_n = len(sixteenth_notes_dur)

    # Average of notes that do not coincide with a given usul
    avg_miss_dur = round(
        (avg_sixteenth_notes_dur - hits_n / sixteenth_notes_n * avg_hits_dur) /
        (1 - (hits_n / sixteenth_notes_n)), 2
    )

    avg_miss_dur_eighth_grid = round(
        (avg_eighth_notes_dur - hits_n / eighth_notes_n * avg_hits_dur) /
        (1 - (hits_n / eighth_notes_n)), 2
    )

    results_dict = {
        'ON usul beat': avg_hits_dur,
        'OFF usul beat': avg_miss_dur,
        'OFF usul beat \n(8th grid only)': avg_miss_dur_eighth_grid
    }

    plt.figure(figsize=(6, 3))
    colors = sns.color_palette("colorblind", 8)

    plt.title(f"""average note duration (in quarter notes \n vs. usul alignment \n
                (disregarding downbeats on 'one')""", fontsize=14)
    plt.bar(*zip(*results_dict.items()), color=colors)
    plt.ylabel('Average duration', fontsize=13)
    plt.xticks(fontsize=13)

    save_dir = '../generated_plots/usul_strokes_v_note_duration'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(os.path.join(save_dir, 'tails_of_measures.pdf'), format='pdf', bbox_inches='tight')
    plt.close()








def gen_plot_per_makam_v_usul(df: pd.DataFrame, makam: str, usul: str, plots_dir: str):
    # Creates directory if it does not exist
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    score_to_outline = {}
    title_measure_ps = df[['title', 'measure', 'pitch_space', 'symbtr_name']]

    # Gets average outline for each measure
    for score, measure_v_pitch in title_measure_ps.groupby(['title']):
        score_avg_outline = [pitch.mean().pitch_space for measure, pitch in 
                             measure_v_pitch.groupby(['measure'])]
        score_to_outline[score] = score_avg_outline
    
    # Resample data pairs and turn them into np.array for plotting
    all_outlines_resampled = np.array([resample(outline, 40) for outline in score_to_outline.values()])
    all_outlines_median = np.quantile(all_outlines_resampled, 0.50, axis=0)
    all_outlines_quartile_1 = np.quantile(all_outlines_resampled, 0.25, axis=0)
    all_outlines_quartile_3 = np.quantile(all_outlines_resampled, 0.75, axis=0)

    # Creates a makam object to display its information
    makam_ = Makam(ascii_name=makam)

    # Plotting code
    scores_n = len(all_outlines_resampled)
    plt.figure(figsize=(10, 6))
    plt.title(f"usul: {usul.lower()}, makam: {makam.lower()}, direction:{makam_.direction.lower()} ({scores_n})", fontsize=15)
    plt.plot(all_outlines_median, c='C0')
    plt.plot(all_outlines_quartile_1, '-.', c='C0')
    plt.plot(all_outlines_quartile_3, '-.', c='C0')
    plt.yticks(ticks=list(makam_.koma_to_symbtr.keys()),
               labels=[(f"{x['special_role']:} " if x['special_role'] !=  None else '') + x['symbtr_name'][:-2] for x in makam_.koma_to_symbtr.values()], fontsize=13)
    plt.savefig(os.path.join(plots_dir, f"{makam}_{usul}_{makam_.direction}_{scores_n}_occ.pdf"), format='pdf', bbox_inches='tight')
    plt.close()








def gen_score_plot_per_measure(df: pd.DataFrame, makam: str, usul: str, plots_dir: str):
    makam_ = Makam(ascii_name=makam)
    title_measure_ps = df[['title', 'measure', 'pitch_space', 'symbtr_name']]
    grouped_by_score = dict(list(title_measure_ps.groupby(['title'])))
    all_present_pitches = set()
    measure_v_avg_per_score = {}
    measure_v_histogram_per_score = {}
    overall_histogram_per_score = {}
    overall_pitch_space_to_symbtr_name_per_score = {}


    # Loops through all items grouped by score and accumulates all note occurences per measure
    # to a short time histogram of usul/makam coincidental onsets
    for score, measure_v_pitch in grouped_by_score.items():
        overall_pitch_space_to_symbtr_name_per_score[score] = dict(zip(measure_v_pitch.pitch_space, measure_v_pitch.symbtr_name))

        for measure, pitch in measure_v_pitch.groupby(['measure']):
            entry = measure_v_avg_per_score.get(score, {})
            mes_list = entry.get('mes_list', [])
            pitch_list = entry.get('pitch_list', [])
            
            mes_list.append(measure)
            pitch_list.append(pitch.mean()['pitch_space'])
            
            entry['mes_list'] = mes_list
            entry['pitch_list'] = pitch_list
            measure_v_avg_per_score[score] = entry

            histo_entry = {}

            for pitch in pitch['pitch_space'].to_list():
                histo_entry[pitch] = histo_entry.get(pitch, 0) + 1
                all_present_pitches.add(pitch)

            histo_per_measure_for_score = measure_v_histogram_per_score.get(score, {})
            histo_per_measure_for_score[measure] = histo_entry

            measure_v_histogram_per_score[score] = histo_per_measure_for_score
        
        overall_histogram_per_score[score] = sum([Counter(dict_) for _, dict_ in measure_v_histogram_per_score[score].items()], Counter())

    # Plotting code
    for title, score in list(measure_v_avg_per_score.items()):
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gridspec = fig.add_gridspec(2, 3)
        fig_ax1 = fig.add_subplot(gridspec[:, :-1])
        fig_ax1.set_title('outline per measure', fontsize=20)

        plt.suptitle(f"{title}, usul: {usul.lower()}, makam: {makam.lower()}, direction: {makam_.direction.lower()}", fontsize=20)

        ticks = list(overall_histogram_per_score[title].keys())
        labels = []

        for koma in ticks:
            entry = makam_.koma_to_symbtr.get(koma, None)

            if entry:
                # Check if current note has a special role in the 'scale'
                if entry['special_role'] != None:
                    labels.append(entry['special_role'].upper() + ': ' + entry['symbtr_name'][:-2])
                else:
                    labels.append(entry['symbtr_name'][:-2])
            else:
                labels.append(overall_pitch_space_to_symbtr_name_per_score[title][koma][:-2])
        
        plt.yticks(ticks=ticks, labels=labels)
        plt.tick_params(axis='both', which='major', labelsize=13)

        # Assigns a different color to notes with special roles (root, dominant, leading_tone)
        # and plot them
        for measure, entries in measure_v_histogram_per_score[title].items():
            for pitch, value in entries.items():
                entry = makam_.koma_to_symbtr.get(pitch, None)

                if entry:
                    if entry['special_role'] != None:
                        if entry['special_role'] == 'root':
                            plt.plot(measure, pitch, 'o', color='#EC2049', markersize=value*5, alpha=0.9)
                        if entry['special_role'] == 'leading_tone':
                            plt.plot(measure, pitch, 'o', color='#F26B38', markersize=value*5, alpha=0.9)
                        if entry['special_role'] == 'dominant':
                            plt.plot(measure, pitch, 'o', color='#A7226E', markersize=value*5, alpha=0.9)
                    else:
                        plt.plot(measure, pitch, 'o', c='C0', markersize=value*5, alpha=0.7)
        
        plt.plot(score['mes_list'], score['pitch_list'], '-.')

        # Plots the overall histogram contour
        fig_ax2 = fig.add_subplot(gridspec[:, -1])
        fig_ax2.set_title('overall histogram', fontsize=20)
        
        bag_of_values = []

        for value, occ in overall_histogram_per_score[title].items():
            for idx in range(occ):
                bag_of_values.append(value)
        
        bins = np.arange(min(bag_of_values) * 2, max(bag_of_values) * 2 + 1) * 0.5

        plt.hist(np.array(bag_of_values), orientation='horizontal', bins=bins)
        plt.yticks(ticks=ticks, labels=['' for x in labels])
        plt.tick_params(axis='both', which='major', labelsize=13)

        # Save plot
        save_dir = os.path.join(plots_dir, f"{makam}_v_{usul}_{makam_.direction}_{len(df.title.unique())}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(os.path.join(save_dir, f'{title}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
    







def gen_usul_visitations_plots(alignment_dict: dict, usuller: str, plots_dir: str):
    for usul in tqdm.tqdm(usuller, desc='Generation usul visitations plot'):
        plots_dir = f"../generated_plots/usul_visitations/{usul}"

        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        keys_with_usul = [makam_v_usul for makam_v_usul in alignment_dict.keys() if makam_v_usul[1] == usul]
        df_list = [alignment_dict[makam_v_usul]['song_to_melodic_outline_df'] for makam_v_usul in keys_with_usul]

        df_full = pd.concat(df_list)[['beat_type', 'duration', 'offset', 'time_sig']]

        for time_signature, df in df_full.groupby('time_sig'):

            # Generates a dict for labeling x axis of the plots
            offsets_to_beat_type = {}

            for beat_v_offset, _ in df.groupby(['beat_type', 'offset']):
                offsets_to_beat_type[beat_v_offset[1]] = beat_v_offset[0]

            offset_vs_dur_sum = {}  # Sum of duration of notes for each offset
            offset_vs_dur_mean = {}  # Mean of duration of notes for each offset
            offset_vs_nr_visits = {}  # Number of visits on each offset regardless of duration

            grouped_by_offset = dict(list(df.groupby(['offset'])))

            for offset, durations in grouped_by_offset.items():
                offset_vs_dur_sum[offset] = durations.duration.sum()
                offset_vs_dur_mean[offset] = durations.duration.mean()
                offset_vs_nr_visits[offset] = len(durations.duration.to_list())
            
            # Plotting code
            colors = sns.color_palette('colorblind', 8)
            plt.title(f"{usul.lower()}({time_signature}): visited usul beats with overall duration in quarters")
            plt.bar(*zip(*offset_vs_dur_sum.items()), width=0.3, color=colors)
            plt.xticks(ticks=list(offset_vs_dur_sum.keys()), 
                    labels=[offsets_to_beat_type[offset] for offset in offset_vs_dur_sum.keys()], fontsize=13, rotation=30)
            plt.savefig(os.path.join(plots_dir, f"{usul.lower()}({time_signature.replace('/', '-')})_duration_sum.pdf"), format='pdf', bbox_inches='tight')
            plt.close()







def gen_avg_note_duration_vs_usul_complete(alignment_dict: dict, plots_dir: str):
    eighth_notes_dur = []
    sixteenth_notes_dur = []
    all_notes_dur = []
    hits_dur = []

    for makam_usul_key in tqdm.tqdm(alignment_dict.keys()):
        df = alignment_dict[makam_usul_key]['song_to_melodic_outline_df']
        # df = df[df.offset != 0.0]

        hits_dur += df.duration.to_list()
        usul_visitations_n = df.shape[0]
        time_signature_occ_count = Counter()

        for time_signature_candidate, time_signature_df in df.groupby(['time_sig']):
            time_signature_occ_count[time_signature_candidate] = time_signature_df.shape[0]
        
        if not usul_visitations_n:
            continue

        makam_usul_specific_time_signature = time_signature_occ_count.most_common()[0][0]
        numerator = makam_usul_specific_time_signature.split('/')[0]
        denominator = makam_usul_specific_time_signature.split('/')[1]
        usul_ = Usul(makam_usul_key[1], makam_usul_specific_time_signature)
        eighth_in_measure_n = int(numerator) * (2 if denominator == '4' else 1)
        bar_length = eighth_in_measure_n * 0.5
        
        # Counts all notes that happen on a sixteenth vs eighth grid
        eighth_notes_count = 0
        sixteenth_notes_count = 0

        for score in alignment_dict[makam_usul_key]['scores_dict'].values():
            score_makam_part = score.parts[0].flat.getElementsByClass(music21.note.Note)

            eighth_notes_dur += [note.duration.quarterLength for note in
                                score_makam_part if note.offset % 1 in [0, 0.5]]
            
            sixteenth_notes_dur += [note.duration.quarterLength for note in
                                    score_makam_part if note.offset % 1 in [0, 0.25, 0.5, 0.75]]
            
            all_notes_dur += [note.duration.quarterLength for note in score_makam_part]
        
    avg_eighth_notes_dur = round(np.mean(eighth_notes_dur), 2)
    avg_sixteenth_notes_dur = round(np.mean(sixteenth_notes_dur), 2)
    avg_hits_dur = round(np.mean(hits_dur), 2)
    hits_n = len(hits_dur)
    eighth_notes_n = len(eighth_notes_dur)
    sixteenth_notes_n = len(sixteenth_notes_dur)

    # Average of notes that do not coincide with a given usul
    avg_miss_dur = round(
        (avg_sixteenth_notes_dur - hits_n / sixteenth_notes_n * avg_hits_dur) /
        (1 - (hits_n / sixteenth_notes_n)), 2
    )

    avg_miss_dur_eighth_grid = round(
        (avg_eighth_notes_dur - hits_n / eighth_notes_n * avg_hits_dur) /
        (1 - (hits_n / eighth_notes_n)), 2
    )

    results_dict = {
        'ON usul beat': avg_hits_dur,
        'OFF usul beat': avg_miss_dur,
        'OFF usul beat \n(8th grid only)': avg_miss_dur_eighth_grid
    }

    plt.figure(figsize=(6, 3))
    colors = sns.color_palette("colorblind", 8)

    plt.title(f"average note duration (in quarter notes \n vs. usul alignment \n", fontsize=14)
    plt.bar(*zip(*results_dict.items()), color=colors)
    plt.ylabel('Average duration', fontsize=13)
    plt.xticks(fontsize=13)

    save_dir = '../generated_plots/usul_strokes_v_note_duration'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(os.path.join(save_dir, 'measures.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
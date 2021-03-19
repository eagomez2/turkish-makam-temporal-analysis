import os
import sys
import functools
import json
import glob
import music21
import unidecode
import pandas as pd
import xml.etree.ElementTree as XMLElementTree
from pprint import pprint
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple


# Custom data types
ScoreType = music21.stream.Score


class NoteMapper:
    """ Helper class for mapping notes from music21 notation to koma53. 

    IMPORTANT: This class is a hierarchical arrangement based on the txt files provided
    in the SymbTr dataset and the most common koma53 values for each not that does not necessarily 
    represent the exact koma53 value of each note since this is not fixed in all cases.
    The usage of this mapping is only for visualization purposes and SHOULD NOT BE CONSIDERED GROUND TRUTH.
    """
    
    # Synthethic koma53 values generated from the txt files of the SymbTr dataset
    _synth_koma53_dict_file = os.path.join(os.path.dirname(__file__), 'synth-koma53.json')
    
    def __init__(self):
        """ Initialization routine. """
        return
                 

    @functools.cached_property
    def _music21_to_koma53_dict(self) -> dict:
        """ Retrieves the synth-koma53 dictionary in an adequate format to be map music21 note names directly and caches it if necessary. 
        
        Returns:
            Formatted music21 to koma53 dict.
        """
        koma53_json = json.load(open(self._synth_koma53_dict_file))
        _music21_to_koma53_dict = {f"{v['note']}{v['octave']}{v['accidental']}": (v['koma53'], k) for k, v in koma53_json.items()}

        return _music21_to_koma53_dict



    def get_symbtr_from_m21(self, base_name: str='C', octave: int=6, accidental: str='slash-flat') -> Tuple[int, str]:
        """ Maps a music21 note name to a (koma53, note_key_idx) tuple. 
        
        Args:
            base_name: Pitch class string (without octave and accidental).
            octave: Octave number.
            accidental: Accidental string as identified in music21.
        
        Returns:
            (koma53, note_key_idx) tuple containing the estimate of koma53 value and the original dictionary key value.

        Raises:
            KeyError: If note is not found in the dictionary.
        """
        note = (base_name, octave, accidental)

        try:
            koma53, note_key_idx = self._music21_to_koma53_dict[f"{note[0]}{note[1]}{note[2]}"]
        except KeyError as e:
            print (f"Note {note} not found in the dicctionary: {e}")
            sys.exit(1)
            # koma, symbtr_name = 999999, "NaN"
        else:
            return koma53, note_key_idx


class ScoreSet:
    """A selected set of scores that are filtered based on makamlar and usuller lists."""

    def __init__(self, dir: str='', makamlar: list=['*'], usuller: list=['*']):
        """ Initialization routine. 
        
        Args:
            dir: Path to the directory where .xml files are stored.
            makamlar: List containing the makamlar list to be selected. '*' can be used as wild card.
            usuller: List containing the usuller to be selected. '*' can be used as wild card.
        """
        super().__init__()
        self.dir = dir
        self.makamlar = makamlar
        self.usuller = usuller


    @property
    def score_files(self) -> list:
        """ Retrieves the path of all selected scores as a list. 

        Returns:
            List of all selected scores.
        """
        makam_filters = [os.path.join(self.dir, f'{makam}--*.xml') for makam in self.makamlar]
        usul_filters = [os.path.join(self.dir, f'*--{usul}--*.xml') for usul in self.usuller]
        
        scores_by_makam = []
        scores_by_usul = []

        for makam_filter in makam_filters:
            scores_by_makam.extend(glob.glob(makam_filter))

        for usul_filter in usul_filters:
            scores_by_usul.extend(glob.glob(usul_filter))

        return list(set(scores_by_makam).intersection(scores_by_usul))

    
    @classmethod
    def _strip_key_signature(cls, score_file: str):
        """ Strip the key signature of a score. 
        
        By stripping the key signature of a score, scores from music traditions different
        than Western can be correctly parsed by music21.

        Args:
            score_file: Path of the score file to be parsed.
        
        Returns:
            new_score: music21.stream.Score without key signature.
        """
        xml_element_tree = XMLElementTree.parse(score_file)
        root = xml_element_tree.getroot()

        for attr in root.iter('attributes'):
            if attr.find('key'):
                attr.remove(attr.find('key'))
            
        new_score_file = f'{os.path.splitext(score_file)[0]}--tmp-nokey.xml'
        xml_element_tree.write(new_score_file)
        new_score = music21.converter.parse(new_score_file)
        os.remove(new_score_file)

        return new_score

    
    @functools.cached_property
    def scores(self) -> List[ScoreType]:
        """ Retrieves all the selected scores as instances of the music21.stream.Score class. 

        Returns:
            scores: A list containing all music21.stream.Score objects corresponding to the selected scores.
        """
        
        scores = []
        unparsable_scores = []

        # Strip key signatures if necessary to avoid forbidden accidentals present in Turkish makam.
        for score_file in tqdm(self.score_files, desc='Parsing and sanitizing scores'):
            try:
                score = music21.converter.parse(score_file)
            except Exception:
                try:
                    score = self._strip_key_signature(score_file)
                except Exception as e:
                    print(f'{score_file} could not be parsed: {e}')
                    unparsable_scores.append(score_file)
                    continue
            
            scores.append(score)
            
        if len(unparsable_scores) > 0:
            print (f"\n{len(unparsable_scores) / len(self.score_files) * 100}% ({len(unparsable_scores)}/{len(self.score_files)}) scores could not be parsed.")
        else:
            print ("\nAll scores were succcessfully parsed.")
        
        return scores


    def _get_file_ranking(self, criterion=None, top_n=None):
        """ Counts the amount of available scores based on a specific criterion such as what makam or usul
        is the most commonly found within the selected list of scores.
        
        Args:
            criterion: Lambda function to be used as criterion to qualify the scores.
            top_n: Maximum number of ranking positions to be included.

        Returns:
            List of tuples containing the categories and the amount of items per each sorted in descending order.
        """
        category_count = []

        for item in self.score_files:
            val = criterion(item)
            category_count.append(val)
        
        return Counter(category_count).most_common(top_n)


    def get_usul_ranking(self, top_n=None):
        """ Returns how many scores per usul are present in the selected scores in descending order. 
        
        Args:
            top_n: Number of ranking positions to be included.

        Returns:
            List of tuples containing the usul and how many scores contain this particular usul in the collection
            ranked in descending order.
        """
        criterion = lambda score_file: os.path.basename(score_file).split('--')[2]
        return self._get_file_ranking(criterion=criterion, top_n=top_n)

    
    def get_makam_ranking(self, top_n=None):
        """ Returns how many scores per makam are present in the selected scores in descending order.

        Args:
            top_n: Number of ranking positions to be included.

        Returns:
            List of tuples containing the makam and how many scores contain this particular makam in the collection
            ranked in descending order.
        """
        criterion = lambda score_file: os.path.basename(score_file).split('--')[0]
        return self._get_file_ranking(criterion=criterion, top_n=top_n)
    

    @functools.cached_property
    def consistent_scores(self) -> List[ScoreType]:
        """ Returns all scores in this ScoreSet that have:
        
        1. Exactly one time signature.
        2. Exactly one text expression specifying makam and usul.
        
        Returns:
            A list of all music21.stream.Score scores matching the given criteria.
        """
        included_scores, filtered_scores = self._apply_score_consistency_filter()
        self.inconsistent_scores = filtered_scores
        return included_scores
    

    @functools.cached_property
    def inconsistent_scores(self)  -> List[ScoreType]:
        """ Returns all scores in this ScoreSet that DO NOT have:

        1. Exactly one time signature.
        2. Exactly one text expression specifying makam and usul.
        
        Returns:
            A list of all music21.stream.Score scores NOT matching the given criteria.
        """
        included_scores, filtered_scores = self._apply_score_consistency_filter()
        self.consistent_scores = included_scores
        return filtered_scores


    def _apply_score_consistency_filter(self) -> Tuple[List[ScoreType], List[ScoreType]]:
        """ 
        Returns two list of scores separated according to the following criteria:

        1. Exactly one time signature.
        2. Exactly one text expression specifying makam and usul.
        
        If a score could not be parsed, it will not be included on any list and a warning will be printed.
        
        Returns:
            A tuple of two lists consisting of all music21.stream.Score matching the criteria and a list of
            all music21.stream.Score NOT matching the criteria.
        """
        unfiltered_scores = []
        filtered_scores = []
        
        # Checks for empty ScoreSet
        if len(self.score_files) == 0:
            print("Empty ScoreSet instance.")
            return [], []

        if len(self.scores) == 0:
            print("Empty ScoreSet instance: None of the scores could be successfully parsed.")
            return [], []
        

        # Copies makamlar and usuller to eliminate wild card
        target_usuller = self.usuller.copy()

        try:
            target_usuller.remove('*')
        except ValueError:
            pass

        target_makamlar = self.makamlar.copy()

        try:
            target_makamlar.remove('*')
        except ValueError:
            pass

        
        # Filter scores with more than one time signature/usul/makam
        for score in tqdm(self.scores, desc='Applying score consistency filter...'):
            score_part = score.parts[0]  # Makam scores have melody in part[0]

            text_expressions = [unidecode.unidecode(text_expression.content).lower() for text_expression 
                                in score.flat.getElementsByClass(music21.expressions.TextExpression)]

            time_signatures = [time_signature.ratioString for time_signature 
                               in score.flat.getElementsByClass(music21.meter.TimeSignature)]

            if len(time_signatures) > 1 or len(text_expressions) > 1:
                filtered_scores.append(score)
                continue


            # if ( # TODO: Usul name hardcoded for now, but must be changed to a dict
            #     not(any(usul in text_expressions[0].replace('devr-i hindi', 'devrihindi') for usul in target_usuller) or len(target_usuller) == 0) or
            #     not(any(makam in text_expressions[0] for makam in target_makamlar) or len(target_makamlar) == 0)
            #     ):

            #    # No matching makam or usul and no wild card is provided
            #    print(f'makam/usul: {text_expressions[0]}')
            #    filtered_scores.append(score)
            #    continue

            unfiltered_scores.append(score)

        print(f"{len(unfiltered_scores) / len(self.score_files) * 100:.2f}% ({len(unfiltered_scores)}/{len (self.score_files)}) scores passed all consistency filters.")

        if len(filtered_scores) > 0:
            print(f"{len(filtered_scores)} filtered because of containing more than one time signature or usul/makam.")

        return unfiltered_scores, filtered_scores
    

    @classmethod
    def produce_makam_usul_overlap_data(cls, MUSIC_XML_DIR: str='', makam_list: List[str]=['rast'], usul_list: List[str]=['sofyan']):
        """
        Retrieves scores that match makam and usul for every possible combination of makam_list and usul_list
        Filter the scores that can be parsed AND that are consistent (see ScoreSet.consistent_scores).
        
        Return:
        Dictionary of:
        (makam, usul) ->
                Dictionary of:
                makam,
                usul,
                'scores_dict':  Dictionary of music21 scores
                'song_to_melodic_outline_dict': dictionary of overlap data extracted
                'song_to_melodic_outline_df': dataframe of overlap data extracted
                'timesig_mismatch_report_all': report of the timesignatures per song, SONG VS USUL
                'timesig_mismatch_report_mismatched': report of the timesignatures per song (where signatures NOT matching), SONG VS USUL
        """
        output = {}

        for makam in makam_list:
            for usul in usul_list:
                output[(makam, usul)] = cls._produce_makam_usul_overlap_data_help(MUSIC_XML_DIR=MUSIC_XML_DIR, makam=makam, usul=usul)
        
        return output
    

    @classmethod
    def _produce_makam_usul_overlap_data_help(cls, MUSIC_XML_DIR: str='', makam: str='rast', usul: str='sofyan'):
        """
        Retrieves scores that match makam and usul.
        Filter the scores that can be parsed AND that are consistent (see ScoreSet.consistent_scores).
        A further requirement is that we can find an Usul pattern of the given kind with the given time signature of a particular score.
        
        Return:
            Dictionary of:
            makam,
            usul,
            'scores_dict':  Dictionary of music21 scores
            'song_to_melodic_outline_dict': dictionary of overlap data extracted
            'song_to_melodic_outline_df': dataframe of overlap data extracted
            'timesig_mismatch_report': report of the timesignatures per song (where signatures NOT matching), SONG VS USUL
            
        
        """
        print(f"\nPairing up usul {usul} with makam {makam}")
        scores = ScoreSet(MUSIC_XML_DIR, makamlar=[makam], usuller=[usul])  # Find matching ScoreSet
        sound_scores = scores.consistent_scores  # Filter consistent scores
        
        # Can optionally be printed to see for scores that do not conform to our consistency guideline
        odd_scores = scores.inconsistent_scores

        melody_overlap_scores = {}
        melody_usul_data = {}
        timesig_mismatch_report = {}

        for score, idx in zip(sound_scores, range(len(sound_scores))):
            title = score.getElementsByClass(music21.metadata.Metadata)[0].title
            composer = score.getElementsByClass(music21.metadata.Metadata)[0].composer
            
            try:
                # score_time_sig = score.parts[0].getElementsByClass(music21.stream.Measure)[0].getElementsByClass(music21.meter.TimeSignature)[0]
                score_time_sig = score.flat.getElementsByClass(music21.meter.TimeSignature)[0].ratioString
            except:
                print (f"No time signature found: {title}")  # Overlaps previous filter
                continue
                
            # score_time_sig = f"{score_time_sig.numerator}/{score_time_sig.denominator}"
            
            try:
                usul_obj = Usul(usul, time_signature=score_time_sig)
            except Exception as e:
                print(e)
                timesig_mismatch_report[title] = (score_time_sig, "Usul not available")
                continue
                
            # Retrieve usul information
            usul_measure = usul_obj.to_measure()
            melody_usul_track = []

            usul_offsets_lyrics = dict((usul_note.offset, usul_note.lyric) for usul_note in usul_measure.notes)
            usul_track = usul_obj.create_measures_like(score)

            melody_track = score.parts[0].getElementsByClass(music21.stream.Measure)

            # Scan melody and produce overlap
            for measure in melody_track:
                melody_usul_measure = []
                
                for note in measure.notes:
                    if note.offset in usul_offsets_lyrics.keys():
                        note.style.color = 'red'
                        melody_usul_measure.append({'offset': note.offset, 
                                                    'offset_abs': measure.offset + note.offset,
                                                    'note': note, 
                                                    'beat_type': usul_offsets_lyrics[note.offset]})
                    else:
                        note.style.color = 'black'

                melody_usul_track.append(melody_usul_measure)

            melody_usul_data[title] = {'title': title,
                                       'composer': composer,
                                       'time_sig': score_time_sig,
                                       'melody_track': melody_usul_track
                                      }

            aggregated_score = music21.stream.Score()
            aggregated_score.insert(0, music21.metadata.Metadata(title=title, composer=composer))

            makam_part = music21.stream.Part(id='makam')
            makam_part.append(melody_track)

            usul_part = music21.stream.Part(id='usul')
            usul_part.append (usul_track)

            aggregated_score.append(makam_part)
            aggregated_score.append(usul_part)

            melody_overlap_scores[title] = aggregated_score
        
        if sound_scores:
            print (f"\nAmount time signature mismatches for Makam {makam}, Usul {usul}", len(timesig_mismatch_report.keys()) / len(sound_scores) * 100, '%')
        else:
            print (f"\nNo parsable and consistent scores left after processing.")

        # print ("================================================")

        return {'makam': makam,
                'usul': usul,
                'scores_dict': melody_overlap_scores, 
                'song_to_melodic_outline_dict': melody_usul_data,
                'song_to_melodic_outline_df': cls._melody_usul_data_to_df (melody_usul_data),
                'timesig_mismatch_report': timesig_mismatch_report
               }
    
    
    @classmethod
    def _melody_usul_data_to_df(cls, melody_usul_data: dict=None):
        """ Transforms the data obtained from the overlapped usul and makam into a pd.DataFrame 
        
        Args:
            melody_usul_data: Raw data generated by _produce_makam_usul_overlap_data_help.

        Returns:
            pd.DataFrame containing the tabulated information to be plotted.
        """
        list_rep = []
        note_mapper = NoteMapper()

        for title, song_data in melody_usul_data.items():
            composer = song_data['composer']
            score_timesig = song_data['time_sig']

            for measure, measure_nr in zip(song_data['melody_track'], range(len(song_data['melody_track']))):
                for note in measure:
                    beat_type = note['beat_type']
                    offset = note['offset']
                    offset_abs = note['offset_abs']
                    duration = note['note'].quarterLength
                    p = note['note'].pitch
                    pitch_rep_m21 = (p.name[0], int(p.implicitOctave), p.accidental.name if p.accidental else '')
                    pitch_name = f"{pitch_rep_m21[0]}{pitch_rep_m21[1]}{pitch_rep_m21[2]}"
                    koma53, symbtr_name = note_mapper.get_symbtr_from_m21 (base_name = p.name[0], octave = int (p.implicitOctave), accidental = p.accidental.name if p.accidental else '')
                    list_rep.append ([title, composer, score_timesig, measure_nr, offset, offset_abs, beat_type, duration, koma53, pitch_name, symbtr_name])
                
        return pd.DataFrame(list_rep, columns = ["title", "composer", "time_sig", "measure", "offset", "offset_abs", "beat_type", "duration", "pitch_space", "pitch_name", "symbtr_name"])


class Usul:
    """ Represents an usul based on the usuller.json dictionary. """

    _usuller_dict_file = os.path.join(os.path.dirname(__file__), 'usuller.json')
    _usuller_dict = json.load(open(_usuller_dict_file))


    def __init__(self, ascii_name: str='aksaksemai', time_signature: str='*'):
        """ Initialization routine. 

        IMPORTANT: For the scope of this work, only one version of each usul is considered for a particular
        time signature, but there may be more variants that are valid too. If you want to include a different one, 
        please modify usuller.json by adding the variant you would like to contrast with the dataset. If more than
        one variant with the same time signature is to be contrasted with this dataset, this class may need some
        coding adaptations to do it properly. Due backwards compatibility, if no time signature is provided, the 
        first usul in the list will be retrieved.
        
        Args:
            ascii_name: ASCII name of the instance's usul among the available ones in usuller.json.
            time_signature: Time signature of the usul to be retrieved. If set to '*', the first usul
            in the list will be retrieved.

        Raises:
            KeyError: Is the usul is not found in usuller.json.
        """
        super().__init__()
        self.ascii_name = ascii_name
        self._data = self._get_usul_data(self.ascii_name, time_signature)
        self.name = self._data['name']
        self.time_signature = self._data['time_signature']
        self.notes = self._data['notes']


    @classmethod
    def usuller_list(cls):
        """ Returns available usuller as a list. 

        Returns:
            Available usuler list.
        """
        return list(cls._usuller_dict.keys())
    

    @classmethod
    def _get_usul_data(cls, ascii_name: str, time_signature: str):
        """ Returns the data associated with a specific usul.
        
        Args:
            ascii_name: ASCII name of the usul to be retrieved.
        
        Returns:
            dict containing the usul's data.
        """ 
        if ascii_name not in cls._usuller_dict.keys():
            raise KeyError(f'Usul {ascii_name} not found in dictionary')
        else:
            if time_signature == '*':
                print('WARNING: No time_signature was provided and therefore first usul in the list was returned')
                return list(cls._usuller_dict[ascii_name].values())[0]  # first variant if wild card is used
            else:
                root_usul = cls._usuller_dict[ascii_name]
                
                #print (root_usul)
                for usul_variant in root_usul.values():
                    #print (usul_variant)
                    if usul_variant['time_signature'] == time_signature:
                        return usul_variant
                
            raise KeyError(f'Usul {ascii_name} has no variants with {time_signature} time signature')
        
        
    def create_measures_like(self, score: ScoreType) -> List[music21.stream.Measure]:
        """ Create usul measures based on a score.
        
        Args:
            score: Score to be taken as reference.

        Returns:
            List of music21 measures generated based on score.
        """
        measure_cnt = len(score.parts[0].getElementsByClass(music21.stream.Measure))
        
        return [self.to_measure(with_time_signature = (i == 0)) for i in range (measure_cnt)]


    def to_measure(self, labels: bool=True, color: str='black', with_time_signature: bool=True):
        """ Returns a music.stream.Measure object containing the usul. 

        IMPORTANT: Please note that in the case of a Usul instance, this function will return
        a single measure with the usul and its respective time signature compared to a Makam 
        instance where it will return a single measure with the whole makam scale in quarter notes.

        Args:
            labels: If True, each usul note will included a lyric with its label.
            color: A string containing the color used to display the notes.

        Returns:
            measure: A music21.stream.Measure object containing the instance's usul.
        """
        measure = music21.stream.Measure()
        measure.append(music21.clef.PercussionClef())
        
        if with_time_signature:
            measure.append(music21.meter.TimeSignature(self.time_signature))

        for note in self.notes:

            if labels:
                new_note = music21.note.Note(note[0], quarterLength=note[1], lyric=note[2])
            else:
                new_note = music21.note.Note(note[0], quarterLength=note[1])

            new_note.style.color=color
            measure.append(new_note)

        return measure
    

    def show(self, *args):
        """ Mirrors the show() functionality available on music21 to load the usul in a notation application. 
        
        Args:
            *args: Arguments to be pased to music21's show() function.
        
        Returns:
            music21 show() functions taking *args as arguments.
        """
        return self.to_measure().show(*args)

    
class Makam:
    """ Represents a makam based on the makamlar.json dictionary. 
    
    IMPORTANT: Please note that due to music21 accidentals limitation, opposite to usuller, makamlar are stored as .musicxml files.
    The makamlar.json file information is still used to determine the root, dominant and leading_tone of each makam, however
    individual note values are not currently being used.
    """
    _makamlar_dict_file = os.path.join(os.path.dirname(__file__), 'makamlar.json')
    _makamlar_dict = json.load(open(_makamlar_dict_file))
    _makamlar_label_colors = {'root': 'red', 'dominant': 'blue', 'leading_tone': 'green'}


    def __init__(self, ascii_name: str='acemasiran'):
        """ Initialization routine.

        Args:
            ascii_name: ASCII name of the makam to be retrieved.
        """
        super().__init__()
        self.ascii_name = ascii_name
        self._data = self._get_makam_data(self.ascii_name)
        self.name = self._data['name']
        self._score_file = self._data['score']
        self.direction = self._data['direction']

    
    @property
    def notes(self):
        """ Returns all the notes corresponding to a specific makam as music21.note.Note objects. """
        makam_score = music21.converter.parse(os.path.join(os.path.dirname(__file__), self._score_file))
        return makam_score.flat.getElementsByClass(music21.note.Note)
    

    @property
    def koma_to_symbtr(self):
        """
        Returns the notes in the given makam as a dictionary (koma53 -> (symbtr_name, note_name, special_role))
        """
        makam_notes = self._data['notes']
        note_mapper = NoteMapper()
        
        koma_to_symbtr = {}
        
        for note_entry in makam_notes:
            special_role = note_entry[1] if len (note_entry) == 2 else None
            note_string = note_entry[0]
            base_name = note_string[0]
            octave = int (note_string[-1])
            accidental_info = note_string[1:-1]
            
            if accidental_info == '':
                pass
            elif accidental_info == 'b':
                accidental_info = 'flat'
            elif accidental_info == '#':
                accidental_info = 'sharp'
            elif accidental_info == '`':
                accidental_info = 'quarter-flat'
            elif accidental_info.startswith('{') and accidental_info.endswith('}'):
                accidental_info = accidental_info[1:-1]
            else:
                raise ValueError('We forgot something here.')
            koma, symbtr_name = note_mapper.get_symbtr_from_m21(base_name=base_name, octave=octave, accidental=accidental_info)
            
            koma_to_symbtr[koma] = {
                'symbtr_name': symbtr_name,
                'note_name': f"{base_name}{octave}{accidental_info}",
                'special_role': special_role
            }
        
        return koma_to_symbtr
        
        
    @classmethod
    def makamlar_list(cls):
        """ Returns available makamlar as a list. 

        Returns:
            Available makam list.
        """
        return list(cls._makamlar_dict.keys())


    @classmethod
    def _get_makam_data(cls, ascii_name: str):
        """ Returns the data associated with a specific makam.
        
        Args:
            ascii_name: ASCII name of the makam to be retrieved.
        
        Returns:
            dict containing the makam's data.
        """ 
        if ascii_name not in cls._makamlar_dict.keys():
            raise KeyError(f'Makam {ascii_name} not found in dictionary')
        else:
            return cls._makamlar_dict[ascii_name]

    
    def to_measure(self, labels=True, color_code=True):
        """ Returns a music.stream.Measure object containing the makam.

        IMPORTANT: Please note that in the case of a Makam instance, this function will return
        a single measure with the scale in quarter notes that is read from the respective makam's musicxml file
        compared to an Usul instance where it will return a single measure with a time signature that corresponds
        to the specific usul.

        Args:
            labels: If True, each makam note will included a lyric with its 'western-ish' function (root, dominant or leading_tone).
            color_code: If True, root, dominant or leading_tone will be color coded.

        Returns:
            measure: A music21.stream.Measure object containing the instance's makam.
        """
        measure = music21.stream.Measure()
        measure.append(music21.clef.TrebleClef())
        time_signature = music21.meter.TimeSignature(f'{len(self.notes)}/4')
        time_signature.style.hideObjectOnPrint = True
        measure.append(time_signature)

        notes_dict = self._data['notes']
        
        for note, note_ in zip(self.notes, notes_dict):

            if len(note_) > 1:
                if labels:
                    note.lyric = note_[1]
                if color_code:
                    note.style.color = self._makamlar_label_colors[note_[1]]

            measure.append(note)

        return measure


    def show(self, *args, labels=True, color_code=True, metadata=True, **kwargs):
        """ Mirrors the show() functionality available on music21 to load the makam in a notation application. 
        
        Args:
            *args: Arguments to be pased to music21's show() function.
        
        Returns:
            music21 show() functions taking *args as arguments.
        """
        score = music21.stream.Score()

        if metadata:
            metadata = music21.metadata.Metadata(title=self.name, composer=self.direction)
            score.insert(0, metadata)

        measure = self.to_measure(labels=labels, color_code=color_code)
        part = music21.stream.Part()
        part.append(measure)
        score.append(part)

        return score.show(*args, **kwargs) 
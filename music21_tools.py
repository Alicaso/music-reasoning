# --------------------------------------------------------------------------
# AGENT TOOLBOX: CORE PARSING UTILITIES
# --------------------------------------------------------------------------

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from music21 import converter, stream, pitch, interval, note, chord, harmony, roman, key, meter, articulations
import music21.environment
from langchain_community.tools import DuckDuckGoSearchRun


def search_music_concept(query: str) -> str:
    """
    Search for music theory concepts, terms, or definitions using DuckDuckGo.
    
    This tool helps the agent understand unfamiliar music concepts by searching
    for detailed explanations and definitions. It's particularly useful when
    encountering specialized terminology or theoretical concepts.

    Args:
        query (str): A brief description of the music concept to search for.
                     Should be a concise sentence describing what you want to learn about.

    Returns:
        str: A comprehensive explanation of the searched music concept,
             or an error message if the search fails.

    Examples:
        - "what is a plagal cadence"
        - "augmented sixth chord characteristics"
        - "sonata form structure"
    """
    try:
        # Initialize the search tool
        search = DuckDuckGoSearchRun()
        
        # Add "music theory" context to improve search relevance
        enhanced_query = f"music theory {query}"
        
        # Perform the search
        results = search.run(enhanced_query)
        
        if not results or len(results.strip()) < 10:
            return f"Error: No relevant information found for '{query}'. Try rephrasing your search query."
        
        # Process and combine multiple relevant snippets for more comprehensive results
        lines = results.split('\n')
        relevant_snippets = []
        
        # Collect multiple relevant snippets instead of just one
        for line in lines:
            line = line.strip()
            if (len(line) > 30 and 
                ('music' in line.lower() or 'theory' in line.lower() or 
                 any(word in line.lower() for word in query.lower().split()))):
                relevant_snippets.append(line)
        
        # If we found relevant snippets, combine them intelligently
        if relevant_snippets:
            # Remove duplicates while preserving order
            unique_snippets = []
            seen = set()
            for snippet in relevant_snippets:
                if snippet not in seen:
                    unique_snippets.append(snippet)
                    seen.add(snippet)
            
            # Combine up to 3 most relevant snippets for comprehensive coverage
            combined_result = " ".join(unique_snippets[:3])
            
            # If the combined result is still too long, truncate more intelligently
            if len(combined_result) > 800:
                # Try to find a good breaking point near 800 characters
                truncated = combined_result[:800]
                last_period = truncated.rfind('.')
                last_sentence = truncated.rfind('!')
                last_question = truncated.rfind('?')
                
                # Find the best breaking point
                best_break = max(last_period, last_sentence, last_question)
                if best_break > 400:  # Only use if we have at least 400 chars
                    combined_result = truncated[:best_break + 1] + "..."
                else:
                    combined_result = truncated + "..."
            
            return f"Search result for '{query}': {combined_result}"
        
        # Fallback: return the first substantial result if no specific matches
        first_result = results.split('\n')[0].strip()
        if len(first_result) > 600:
            # Find a good breaking point
            truncated = first_result[:600]
            last_period = truncated.rfind('.')
            if last_period > 300:
                first_result = truncated[:last_period + 1] + "..."
            else:
                first_result = truncated + "..."
        
        return f"Search result for '{query}': {first_result}"
        
    except Exception as e:
        return f"Error: Failed to search for '{query}'. Details: {str(e)}"


def parse_kern_string_to_score(kern_data: str) -> Optional[stream.Score]:
    """
    Robustly parses a Humdrum **kern** notation string into a music21 Score object.

    This function explicitly calls the 'humdrum' parser (the correct format key for kern)
    and handles exceptions silently on success. It standardizes the output to always 
    be a Score object for consistent downstream tool processing:
    - If parsing yields a Part (single spine), it wraps it in a new Score.
    - If parsing yields an Opus (multi-piece file, '!!!' separated), it returns the first Score.

    Args:
        kern_data (str): A string containing complete **kern** (Humdrum) notation.

    Returns:
        Optional[stream.Score]: A music21 Score object on success.
                                On failure, prints an error to console and returns None.
    """
    if not kern_data.strip():
        return None

    # Quick sanity check to avoid attempting to parse obvious non-kern inputs
    # (reduces noisy exceptions from the humdrum parser)
    if "**kern" not in kern_data:
        return None

    try:
        # Core parsing step: use 'humdrum' as the format key for kern strings.
        parsed_stream = converter.parse(kern_data, format='humdrum')

        # --- Output Standardization Logic (identical to ABC parser for consistency) ---
        if isinstance(parsed_stream, stream.Score):
            return parsed_stream  # Success
        
        elif isinstance(parsed_stream, stream.Part):
            # Wrap a single Part/Spine in a Score.
            new_score = stream.Score()
            new_score.insert(0, parsed_stream)
            return new_score
            
        elif hasattr(parsed_stream, 'scores') and len(parsed_stream.scores) > 0:
            # This is an Opus object. Return the first score.
            return parsed_stream.scores[0]
            
        else:
            # Handle unexpected parsing results silently.
            return None

    except converter.ConverterException:
        # Handle music21-specific parsing failures silently.
        return None
    except Exception:
        # Handle any other unexpected Python errors silently.
        return None

PITCH_CLASS_MAP = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
]

# ==========================================================================
# TOOL 1: Pitch-Class Identification (Natural Language Output Version)
# ==========================================================================

def get_pitch_classes_in_segment(
    kern_data: str, 
    part_index: int,
    measure_start: int, 
    measure_end: int
) -> str:
    """
    Analyze unique pitch classes (0-11) within a specified part and measure range, and return their names.

    Args:
        kern_data (str): The raw Humdrum/Kern string for the entire score.
        part_index (int): The mandatory 0-based index of the specific part.
        measure_start (int): The starting measure number to analyze (inclusive).
        measure_end (int): The ending measure number to analyze (inclusive).

    Returns:
        str: A single natural language sentence describing the findings
             (e.g., "The pitch classes found in Part 0 from measure 1 to 2 are: [C, D, E, G, A].")
             or a string describing the error (e.g., "Error: Parsing failed.").
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        # Return the error as a natural language string
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    selection_stream = None
    
    try:
        # 2. Validate and Select the mandatory part
        if not (0 <= part_index < len(score.parts)):
            return f"Error: Invalid part_index: {part_index}. Score only has {len(score.parts)} parts (indexed 0 to {len(score.parts)-1})."
        
        target_part = score.parts[part_index]
            
        # 3. Slice the selected part by measure range
        selection_stream = target_part.measures(measure_start, measure_end)
            
        if not selection_stream or selection_stream.duration.quarterLength == 0:
             # Valid query, segment is just empty.
             return f"No pitch classes (notes) were found in Part {part_index} from measure {measure_start} to {measure_end}."

        # 4. Analyze: Get all pitches and unique pitch classes
        all_pitches_in_selection = selection_stream.flat.pitches
        pitch_class_set = {p.pitchClass for p in all_pitches_in_selection}
        sorted_pcs = sorted(list(pitch_class_set))
        
        # 5. Format the output as requested
        if not sorted_pcs:
            # Re-check for empty list after analysis (e.g., segment only had rests)
            return f"No pitch classes (notes) were found in Part {part_index} from measure {measure_start} to {measure_end}."
        
        # Map integers to names using the predefined map
        pc_name_list = [PITCH_CLASS_MAP[pc_int] for pc_int in sorted_pcs]
        name_string = ", ".join(pc_name_list)

        # 6. Report: Return the final natural language sentence
        return (f"The pitch classes found in Part {part_index} from measure {measure_start} to {measure_end} "
                f"are: [{name_string}].")

    except Exception as e:
        return f"Error: An unexpected analysis error occurred: {e}"

# ==========================================================================
# INTERNAL HELPER FUNCTION (for Tool 2)
# ==========================================================================

def _get_event_by_coordinate(
    score: stream.Score, 
    part_idx: int, 
    measure_num: int, 
    event_idx: int, 
    event_label: str = "Event"
) -> Tuple[Optional[note.Note], Optional[str]]:
    """
    Internal helper to robustly retrieve a single Note/Chord object 
    using the 3-part (part, measure, event) coordinate.
    
    Returns:
        (event_object, None) on success.
        (None, error_string) on failure.
    """
    # 1. Validate Part Index
    num_parts = len(score.parts)
    if not (0 <= part_idx < num_parts):
        return None, (f"Error ({event_label}): Invalid part_index: {part_idx}. "
                      f"Score only has {num_parts} parts (indexed 0 to {num_parts - 1}).")
    
    target_part = score.parts[part_idx]

    # 2. Validate Measure Number
    # We use .measure() which correctly fetches by *number*, not list index.
    target_measure = target_part.measure(measure_num)
    if target_measure is None:
        return None, f"Error ({event_label}): Measure {measure_num} not found in Part {part_idx}."

    # 3. Validate Event Index
    # .notes filters for *only* Note and Chord objects, ignoring Rests, Clefs, etc.
    # This is what the user logically means by an "event" to compare.
    events_in_measure = list(target_measure.notes)
    num_events = len(events_in_measure)
    
    if not (0 <= event_idx < num_events):
        # Handle the case where the measure is valid but has 0 events
        if num_events == 0:
             return None, (f"Error ({event_label}): Invalid event_index: {event_idx}. "
                           f"Measure {measure_num} (Part {part_idx}) contains no note/chord events.")
        
        return None, (f"Error ({event_label}): Invalid event_index: {event_idx}. "
                      f"Measure {measure_num} only has {num_events} note/chord events "
                      f"(indexed 0 to {num_events - 1}).")

    # 4. Success
    return events_in_measure[event_idx], None


# ==========================================================================
# TOOL 2: Interval Recognition (Point-to-Point)
# ==========================================================================

def get_interval_between_two_events(
    kern_data: str, 
    part_a_index: int, 
    measure_a_num: int, 
    event_a_index: int,
    part_b_index: int, 
    measure_b_num: int, 
    event_b_index: int
) -> str:
    """
    Compute the directed interval between two events (note or chord).
    
    Each event is located using a 3-part coordinate:
    1. part_index (0-based)
    2. measure_number (1-based, matching the score)
    3. event_index (0-based index of the Note/Chord *within* that measure)
    
    If an event is a Chord, its .bass() (lowest) pitch is used for the calculation.
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Locate Event A
        event_a, error_a = _get_event_by_coordinate(
            score, part_a_index, measure_a_num, event_a_index, "Event A"
        )
        if error_a:
            return error_a # Return the specific, helpful error message

        # 3. Locate Event B
        event_b, error_b = _get_event_by_coordinate(
            score, part_b_index, measure_b_num, event_b_index, "Event B"
        )
        if error_b:
            return error_b # Return the specific, helpful error message

        # 4. Get pitches (using .bass() is robust for both Notes and Chords)
        pitch_a = event_a.bass() 
        pitch_b = event_b.bass()

        # 5. Calculate the interval
        ivl = interval.Interval(pitch_a, pitch_b)

        # 6. Format the natural language report
        ivl_name = ivl.name
        ivl_semitones = ivl.semitones

        # Determine direction
        if ivl_semitones > 0:
            direction = "ascending"
        elif ivl_semitones < 0:
            direction = "descending"
        else:
            direction = "unison"

        # Construct coordinate strings for the report
        coord_a_str = f"P{part_a_index}, M{measure_a_num}, E{event_a_index}"
        coord_b_str = f"P{part_b_index}, M{measure_b_num}, E{event_b_index}"

        return (f"The interval from Event A ({coord_a_str}) to Event B ({coord_b_str}) "
                f"is a {ivl_name} ({direction}), with a distance of {ivl_semitones} semitones.")

    except Exception as e:
        return f"Error: An unexpected analysis error occurred during interval calculation: {e}"

# ==========================================================================
# CHORD TOOL 1: Chord Root Detection
# ==========================================================================

def get_chord_root_at_event(
    kern_data: str, 
    part_index: int, 
    measure_num: int, 
    event_index: int
) -> str:
    """
    Return the conceptual root of the specified event.
    
    - This tool uses chord.root() to find the conceptual root, which is DIFFERENT
      from the .bass() (lowest sounding note) used by the interval tool.
    - Example: For a C-Major chord in first inversion (E-G-C), .bass() returns 'E',
      but this tool (.root()) will correctly identify 'C'.
    - If the event is just a single Note, it returns that note's pitch as the root.
    """
    
    # 1. Parse the music data (via the shared helper function)
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Locate the single event using the exact same helper as Tool 2
        event, error = _get_event_by_coordinate(
            score, part_index, measure_num, event_index, "Event"
        )
        if error:
            return error  # Return the specific coordinate error

        coord_str = f"(P{part_index}, M{measure_num}, E{event_index})"

        # 3. Analyze the event
        if isinstance(event, chord.Chord):
            # This is the main use case. Run the root-finding algorithm.
            root_pitch = event.root()
            return (f"Event {coord_str} is a Chord. "
                    f"The calculated conceptual root is: {root_pitch.name}.")
        
        elif isinstance(event, note.Note):
            # This is a valid edge case. The root of a single note is itself.
            root_pitch = event.pitch
            return (f"Event {coord_str} is a single Note. "
                    f"Its pitch (and root) is: {root_pitch.name}.")
        
        else:
            # This should not be reachable if .notes was used in the helper
            return f"Error: Event {coord_str} is not a recognizable Note or Chord."

    except Exception as e:
        # Catch-all for any other unexpected analysis errors
        return f"Error: An unexpected analysis error occurred during root calculation: {e}"

# ==========================================================================
# ==========================================================================
# CHORD TOOL 2: Chordify (Intelligent Naming)
# ==========================================================================

def get_chord_progression_in_segment(
    kern_data: str, 
    measure_start: int, 
    measure_end: int
) -> str:
    """
    Chordify the full score for the given measure range and return the per-measure chord progression.
    
    - This tool ALWAYS analyzes all parts combined (full score harmony).
    - It intelligently names chords (e.g., "Cm", "G7").
    - If it encounters a monophonic (single-note) event, it returns the 
      note name (e.g., "C") as the symbol for that position.
    - If it encounters a complex/unnamable multi-note cluster, it returns "(Unnamable-A, B, C)" 
      where A, B, C are the actual pitch names in the cluster.
    - Chords within a measure are separated by ", " (comma-space).
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Run Chordify on the ENTIRE score
        chordified_stream = score.chordify()
        
        # 3. Slice the requested measures
        segment_slice = chordified_stream.measures(measure_start, measure_end)

        # 4. Iterate by measure and apply the new intelligent naming logic
        output_segments = []
        has_any_events = False

        for m in segment_slice.getElementsByClass('Measure'):
            chords_in_measure = m.getElementsByClass('Chord')
            chord_names = []
            
            for c in chords_in_measure:
                has_any_events = True
                name = ""
                num_pitches = len(c.pitches)

                if num_pitches == 1:
                    
                    name = c.pitches[0].name
                else:
                    # It's a "real" chord (2+ notes). Try to name it.
                    name = harmony.chordSymbolFigureFromChord(c)
                    if name == "Chord Symbol Cannot Be Identified":
                        # It's a complex cluster, dyad, or other non-namable chord.
                        # Include the actual pitches for agent analysis
                        pitch_names = [p.name for p in c.pitches]
                        name = f"(Unnamable-{', '.join(pitch_names)})"
                
                chord_names.append(name)
            
            
            output_segments.append(", ".join(chord_names))

        # 5. Check if we actually found anything
        if not has_any_events:
             return (f"No chords or notes could be identified in the specified segment "
                     f"(M {measure_start}-{measure_end}).")

        # 6. Build the final lead-sheet string
        final_prog_string = "| " + " | ".join(output_segments) + " |"
        
        
        return (f"Chord progression for measures {measure_start}-{measure_end}: "
                f"{final_prog_string}")
        
    except Exception as e:
        return f"Error: An unexpected error occurred during chordification: {e}"

# ==========================================================================
# CHORD TOOL 4: Roman Numeral Analysis
# ==========================================================================

def get_roman_numeral_analysis_in_segment(
    kern_data: str,
    measure_start: int,
    measure_end: int,
    key_string: str
) -> str:
    """
    Performs Roman numeral analysis on a segment of the score and returns the progression.

    - This tool analyzes all parts combined (full score harmony) within the specified measure range.
    - It first chordifies the score, then analyzes each resulting chord against the provided key.
    - If a chord cannot be analyzed (e.g., it's a single note, a complex cluster, or a dyad),
      it is marked as '(Complex/Ambiguous)'.
    - The analysis for each measure is presented in a format like "| I, V | vi, IV |".

    Args:
        kern_data (str): The raw Humdrum/Kern string for the entire score.
        measure_start (int): The starting measure number to analyze (inclusive).
        measure_end (int): The ending measure number to analyze (inclusive).
        key_string (str): The key context for the analysis (e.g., 'C', 'am', 'Bb').

    Returns:
        str: A natural language string containing the Roman numeral progression for the segment,
             or a string describing an error.
    """
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Validate the Key Context string FIRST
        key_obj = None
        try:
            key_obj = key.Key(key_string)
        except Exception:
            return (f"Error: Invalid key_string provided: '{key_string}'. "
                    f"Music21 could not parse it. Use format like 'C' (major), "
                    f"'cm' (minor), 'Bb', 'g#'.")

        # 3. Run Chordify on the ENTIRE score
        chordified_stream = score.chordify()

        # 4. Slice the requested measures from the chordified stream
        segment_slice = chordified_stream.measures(measure_start, measure_end)

        # 5. Iterate by measure and analyze each chord
        output_segments = []
        has_any_events = False

        for m in segment_slice.getElementsByClass('Measure'):
            chords_in_measure = m.getElementsByClass('Chord')
            roman_numerals_in_measure = []

            for c in chords_in_measure:
                has_any_events = True
                analysis_figure = ""
                try:
                    # Attempt to get the Roman numeral for the current chord
                    rn = roman.romanNumeralFromChord(c, key_obj)
                    analysis_figure = rn.figure
                except roman.RomanNumeralException:
                    # This handles cases where the chord is a single note, a dyad,
                    # a complex cluster, or otherwise unanalyzable in the given key.
                    analysis_figure = "(Complex/Ambiguous)"

                roman_numerals_in_measure.append(analysis_figure)

            output_segments.append(", ".join(roman_numerals_in_measure))

        # 6. Check if we actually found anything to analyze
        if not has_any_events:
            return (f"No chords or notes could be identified in the specified segment "
                    f"(M {measure_start}-{measure_end}) for analysis.")

        # 7. Build the final lead-sheet style string
        final_analysis_string = "| " + " | ".join(output_segments) + " |"

        return (f"Roman numeral analysis in {key_obj.name} for measures {measure_start}-{measure_end}: "
                f"{final_analysis_string}")

    except Exception as e:
        return f"Error: An unexpected error occurred during Roman numeral analysis: {e}"


# ==========================================================================
# INTERNAL HELPER FUNCTION (for Melodic Contour)
# ==========================================================================

def _get_top_pitch(event) -> Optional[pitch.Pitch]:
    """
    Internal helper to get the perceived "melody" pitch from an event.
    For a Note, returns its pitch.
    For a Chord, returns its HIGHEST pitch (the soprano line).
    """
    if isinstance(event, note.Note):
        return event.pitch
    elif isinstance(event, chord.Chord):
        if event.pitches:
            # Pitches are stored sorted from low to high, so [-1] is the highest.
            return event.pitches[-1]
    return None # Should not happen if we only feed it notes/chords

def _format_semitone(s: int) -> str:
    """Helper to format integers into the user's requested [+N, -N, 0] format."""
    if s > 0:
        return f"+{s}"
    return str(s) # Automatically handles 0 and negative numbers

# ==========================================================================
# MELODIC TOOL 1: Get Melodic Contour
# ==========================================================================

def get_melodic_contour_of_part(
    kern_data: str, 
    part_index: int,
    measure_start: int, 
    measure_end: int
) -> str:
    """
    Compute the melodic contour (semitone steps between consecutive events) for a single part within a measure range.
    
    - This tool requires a mandatory part_index.
    - It intelligently handles chords by selecting their HIGHEST pitch (the soprano/melody line)
      as the melodic event for that timestep.
    - It skips rests and calculates the interval between consecutive sounding events.
    - Returns a list of semitone steps, e.g., [+2, -1, 0, +7].
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Validate and Select the mandatory part
        num_parts = len(score.parts)
        if not (0 <= part_index < num_parts):
            return (f"Error: Invalid part_index: {part_index}. "
                    f"Score only has {num_parts} parts (indexed 0 to {num_parts - 1}).")
        
        target_part = score.parts[part_index]
        
        # 3. Process each measure separately to add bar separators
        measure_contours = []
        total_events = 0
        
        for measure_num in range(measure_start, measure_end + 1):
            # Get events for this specific measure
            measure_obj = target_part.measure(measure_num)
            if measure_obj is None:
                continue
                
            # Get all sounding events (Notes/Chords) in this measure
            events_in_measure = list(measure_obj.flatten().notes)
            
            if len(events_in_measure) < 2:
                # Not enough events in this measure for contour
                measure_contours.append("[]")
                continue
            
            # Extract the "Top Line" pitches for this measure
            top_line_pitches = [_get_top_pitch(e) for e in events_in_measure if _get_top_pitch(e) is not None]
            
            if len(top_line_pitches) < 2:
                # Not enough valid pitches in this measure
                measure_contours.append("[]")
                continue
            
            # Calculate intervals between consecutive top-line pitches in this measure
            measure_contour_semitones = []
            for i in range(len(top_line_pitches) - 1):
                pitch_a = top_line_pitches[i]
                pitch_b = top_line_pitches[i+1]
                ivl = interval.Interval(pitch_a, pitch_b)
                measure_contour_semitones.append(ivl.semitones)
            
            # Format this measure's contour
            measure_contour_formatted = [_format_semitone(s) for s in measure_contour_semitones]
            measure_contours.append(f"[{', '.join(measure_contour_formatted)}]")
            total_events += len(measure_contour_semitones)
        
        if total_events < 1:
            return (f"Insufficient notes (< 2) in Part {part_index} (M {measure_start}-{measure_end}) "
                    f"to calculate a contour.")

        # 4. Format the output with bar separators
        final_string = " | ".join(measure_contours)
        
        return (f"Melodic contour for Part {part_index} (Measures {measure_start}-{measure_end}): "
                f"{final_string}")

    except Exception as e:
        return f"Error: An unexpected analysis error occurred during contour calculation: {e}"

# ==========================================================================
# KEY TOOL 1: Key Signature Extraction
# ==========================================================================

def extract_all_key_signatures(kern_data: str) -> str:
    """
    Extract all explicitly notated key signatures with their measure ranges, and handle scores that start without an explicit key signature.
    """
    
    # 1. Parse
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Get all KS objects, sorted by offset
        all_ks_objects = score.flatten().getElementsByClass('KeySignature')
        
        change_events = [] 
        last_sharps_count = None 

        # 3. De-duplicate to find only points of CHANGE
        for ks in all_ks_objects:
            current_sharps_count = ks.sharps
            # Ensure measureNumber is not None (can happen for elements outside measures)
            measure_num = ks.measureNumber if ks.measureNumber is not None else 1
            
            if current_sharps_count != last_sharps_count:
                major_name = ks.asKey('major').name
                minor_name = ks.asKey('minor').name
                full_name_str = f"{major_name} (or {minor_name})"
                
                # Avoid adding duplicates at the same measure (e.g., if offset maps oddly)
                if not change_events or change_events[-1]["measure"] != measure_num:
                    change_events.append({
                        "measure": measure_num,
                        "name": full_name_str,
                        "sharps": current_sharps_count
                    })
                last_sharps_count = current_sharps_count

        
        # If any signatures were found, BUT the first one does NOT start at measure 1,
        # we must manually prepend the "default" (un-notated) key signature.
        if change_events and change_events[0]["measure"] > 1:
            default_key_name = "C major (or a minor)"
            default_event = {
                "measure": 1, # It applies from measure 1
                "name": default_key_name,
                "sharps": 0
            }
            # Only add if the late-starting key isn't *also* C major
            if change_events[0]["sharps"] != 0:
                 change_events.insert(0, default_event)
            else:
                 # The key *is* C major, it just was written late. Fix the measure number.
                 change_events[0]["measure"] = 1


        # 5. Format the output based on the (now corrected) list
        if not change_events:
            return "No key signature is explicitly notated in the score (defaults to C major / a minor)."
        
        elif len(change_events) == 1:
            # This case is now safe: it means only one signature exists AND it started at measure 1.
            evt = change_events[0]
            return (f"The notated key signature is: {evt['name']} "
                    f"({evt['sharps']} sharps/flats) from beginning to end.")
        
        else:
            # This case now correctly handles the "default -> change" scenario
            report_parts = ["A sequence of key signature changes was found:"]
            for i in range(len(change_events)):
                evt = change_events[i]
                start_measure = evt['measure']
                
                # Special handling for our manually inserted default name
                if evt['sharps'] == 0 and start_measure == 1 and change_events[i+1]['measure'] > 1:
                     name_str = f"No signature notated (defaults to {evt['name']}) ({evt['sharps']} sharps/flats)"
                else:
                     name_str = f"{evt['name']} ({evt['sharps']} sharps/flats)"
                
                end_measure_str = ""
                if i + 1 < len(change_events):
                    end_measure = change_events[i+1]['measure'] - 1
                    if end_measure < start_measure:
                         end_measure = start_measure # Handle changes in same measure
                    end_measure_str = f"to measure {end_measure}"
                else:
                    end_measure_str = "to the end"
                
                report_parts.append(f"- From measure {start_measure} {end_measure_str}: {name_str}.")
            
            return " ".join(report_parts)

    except Exception as e:
        return f"Error: An unexpected error occurred during key signature extraction: {e}"

# ==========================================================================
# RHYTHM TOOL 1: Time Signature Extraction
# ==========================================================================

def extract_all_time_signatures(kern_data: str) -> str:
    """
    Extract all explicitly notated time signatures with their measure ranges, and handle scores that start without an explicit time signature.
    """
    
    # 1. Parse
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Get all TS objects, sorted
        all_ts_objects = score.flatten().getElementsByClass('TimeSignature')
        
        change_events = [] 
        last_ts_string = None

        # 3. De-duplicate to find only points of CHANGE
        for ts in all_ts_objects:
            current_ts_string = ts.ratioString
            measure_num = ts.measureNumber if ts.measureNumber is not None else 1
            
            if current_ts_string != last_ts_string:
                 # Avoid adding duplicates at the same measure
                if not change_events or change_events[-1]["measure"] != measure_num:
                    change_events.append({
                        "measure": measure_num,
                        "name": current_ts_string
                    })
                last_ts_string = current_ts_string

        
        # If any signatures were found, BUT the first one does NOT start at measure 1,
        # we must manually prepend an "(Un-notated)" event at measure 1.
        if change_events and change_events[0]["measure"] > 1:
            default_event = {
                "measure": 1,
                "name": "(Un-notated)" # Clear tag for "no signature found"
            }
            change_events.insert(0, default_event)

        # 5. Format the output based on the (now corrected) list
        if not change_events:
            return "No time signature is explicitly notated in the score."
        
        elif len(change_events) == 1:
            evt = change_events[0]
            return f"The notated time signature is: {evt['name']} from beginning to end."
        
        else:
            # This case now correctly handles the "(Un-notated) -> change" scenario
            report_parts = ["A sequence of time signature changes was found:"]
            for i in range(len(change_events)):
                evt = change_events[i]
                start_measure = evt['measure']
                name_str = evt['name']
                
                end_measure_str = ""
                if i + 1 < len(change_events):
                    end_measure = change_events[i+1]['measure'] - 1
                    if end_measure < start_measure:
                        end_measure = start_measure
                    end_measure_str = f"to measure {end_measure}"
                else:
                    end_measure_str = "to the end"
                
                report_parts.append(f"- From measure {start_measure} {end_measure_str}: {name_str}.")
            
            return " ".join(report_parts)

    except Exception as e:
        return f"Error: An unexpected error occurred during time signature extraction: {e}"

# ==========================================================================
# KEY TOOL 2: Key Estimation
# ==========================================================================

def get_key_estimation(
    kern_data: str, 
    measure_start: Optional[int] = None,
    measure_end: Optional[int] = None
) -> str:
    """
    Estimate the key of the entire score (or a specified range) using the Krumhansl-Schmuckler algorithm and report confidence.
    
    This tool analyzes ALL parts combined (as concert pitch) to find the single, 
    unified tonal center for the specified segment.
    
    - If measure_start/end are omitted, it analyzes the ENTIRE score.
    - If measure_start/end are provided, it analyzes ONLY that slice.
    
    It returns the winning key AND the confidence level (correlation coefficient).
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        target_stream = score
        range_desc = "the full score"

        # 2. Slice the stream if a range is provided
        if measure_start is not None and measure_end is not None:
            target_stream = score.measures(measure_start, measure_end)
            range_desc = f"measures {measure_start}-{measure_end}"
            
            # Check if the slice is empty or contains no notes, which would cause .analyze() to fail
            if not target_stream or not list(target_stream.flatten().notes):
                return (f"Error: The specified measure range ({range_desc}) "
                        f"is empty or contains no notes to analyze.")

        # 3. Run the analysis
        # .analyze('key') runs the algorithm on the given stream (or stream slice)
        estimated_key = target_stream.analyze('key')

        # 4. Format the output
        key_name = estimated_key.name
        confidence = estimated_key.correlationCoefficient
        
        # Format the float to 3 decimal places for clarity
        confidence_str = f"{confidence:.3f}"
        
        return (f"Based on pitch analysis of {range_desc}, the highest probability key is: "
                f"{key_name}. (Confidence: {confidence_str})")

    except Exception as e:
        return f"Error: An unexpected error occurred during key estimation: {e}"

# ==========================================================================
# KEY TOOL 3: Windowed Key Estimation
# ==========================================================================

def get_windowed_key_estimation(
    kern_data: str, 
    window_size: Optional[int] = 4,
    step_size: Optional[int] = 1
) -> str:
    """
    Perform sliding-window key estimation over the full score, reporting the key and confidence for each window.
    
    This tool analyzes the TOTAL harmony (all parts combined) at each step.
    It does NOT use chordify; it analyzes the pitches from the raw score slice.
    
    Args:
        kern_data (str): The full kern data string.
        window_size (Optional[int]): The size of the analysis window (in measures).
                                     Defaults to 4, which is standard for a musical phrase.
        step_size (Optional[int]): How many measures to slide the window forward at each step.
                                   Defaults to 1 (overlapping window).
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    if window_size <= 0 or step_size <= 0:
         return "Error: window_size and step_size must be positive integers."

    try:
        # 2. Get the total number of measures from the (representative) first part
        # This is the most reliable way to get the measure count.
        last_measure_num = 0
        if score.parts:
             # Get the number of the final measure in Part 0
             part_measures = score.parts[0].getElementsByClass('Measure')
             if part_measures:
                 last_measure_num = part_measures[-1].number
        
        if last_measure_num < window_size:
             return (f"Error: Score length ({last_measure_num} measures) is shorter "
                     f"than the window size ({window_size} measures).")

        results = [] # Store the string result of each window

        # 3. Iterate using the sliding window
        # Loop from measure 1 up to the last *possible* start of a full window
        for m_start in range(1, (last_measure_num - window_size + 2), step_size):
            m_end = (m_start + window_size) - 1
            
            # This slice (a new Score object) contains all parts for this measure range
            segment_slice = score.measures(m_start, m_end)

            key_name = "NoData"
            confidence_str = "N/A"

            # Check if the slice actually has notes to analyze
            if segment_slice and list(segment_slice.flatten().notes):
                # 4. Analyze the slice (all parts combined)
                estimated_key = segment_slice.analyze('key')
                key_name = estimated_key.name
                confidence = estimated_key.correlationCoefficient
                confidence_str = f"{confidence:.3f}"
            
            results.append(f"[M.{m_start}-{m_end}: {key_name} (Conf: {confidence_str})]")

        if not results:
            return "No analysis windows were generated. Check score and parameters."

        # 5. Format the final sequence string
        final_sequence = ", ".join(results)
        
        return (f"Key analysis sequence (Window={window_size}, Step={step_size}): "
                f"{final_sequence}")

    except Exception as e:
        return f"Error: An unexpected error occurred during windowed key estimation: {e}"

# ==========================================================================
# RHYTHM TOOL 2: Duration Calculator
# ==========================================================================

def calculate_duration_of_segment(
    kern_data: str, 
    part_index: int,
    start_measure_num: int, 
    start_event_index: int,
    end_measure_num: int, 
    end_event_index: int
) -> str:
    """
    Compute the total duration (in quarter notes) from the start of the start event to the end of the end event.
    Use the sum of the measure container absolute offset and the event's relative offset to obtain accurate timing.
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Validate Part Index and get the target Part (original, nested)
        num_parts = len(score.parts)
        if not (0 <= part_index < num_parts):
            return (f"Error: Invalid part_index: {part_index}. "
                    f"Score only has {num_parts} parts (indexed 0 to {num_parts - 1}).")
        
        target_part = score.parts[part_index] # This is the stream we query

        # 3. --- THIS IS THE CORRECTED LOGIC ---
        # Get Start Event AND its absolute offset
        
        # 3a. Find the Start Measure *container*
        start_measure_obj = target_part.measure(start_measure_num)
        if start_measure_obj is None:
            return f"Error (Start Event): Measure {start_measure_num} not found in Part {part_index}."
            
        # 3b. Find the Event *within* that measure
        start_events_list = list(start_measure_obj.notes)
        if not (0 <= start_event_index < len(start_events_list)):
             return (f"Error (Start Event): Invalid event_index: {start_event_index}. "
                     f"Measure {start_measure_num} only has {len(start_events_list)} note/chord events.")
        start_event = start_events_list[start_event_index]

        # 3c. Calculate the absolute start time
        start_measure_abs_offset = start_measure_obj.getOffsetBySite(target_part)
        start_time_abs = start_measure_abs_offset + start_event.offset

        # 4. Get End Event AND its absolute offset
        
        # 4a. Find the End Measure *container*
        end_measure_obj = target_part.measure(end_measure_num)
        if end_measure_obj is None:
            return f"Error (End Event): Measure {end_measure_num} not found in Part {part_index}."

        # 4b. Find the Event *within* that measure
        end_events_list = list(end_measure_obj.notes)
        if not (0 <= end_event_index < len(end_events_list)):
             return (f"Error (End Event): Invalid event_index: {end_event_index}. "
                     f"Measure {end_measure_num} only has {len(end_events_list)} note/chord events.")
        end_event = end_events_list[end_event_index]
        
        # 4c. Calculate the absolute end time (including the event's duration)
        end_measure_abs_offset = end_measure_obj.getOffsetBySite(target_part)
        end_time_abs = end_measure_abs_offset + end_event.offset + end_event.duration.quarterLength

        # 5. Calculate total duration
        total_duration = end_time_abs - start_time_abs

        if total_duration <= 0:
            return (f"Error: End Event (occurs at {end_measure_abs_offset + end_event.offset}) "
                    f"does not occur at or after Start Event (occurs at {start_time_abs}). Duration must be positive.")

        # 6. Report
        start_coord_str = f"(P{part_index}, M{start_measure_num}, E{start_event_index})"
        end_coord_str = f"(P{part_index}, M{end_measure_num}, E{end_event_index})"

        return (f"The total duration from the start of {start_coord_str} "
                f"to the end of {end_coord_str} is: {total_duration} quarter notes.")

    except Exception as e:
        return f"Error: An unexpected error occurred during duration calculation: {e}"


# ==========================================================================
# INTERNAL HELPER (FOR STATS TOOLS)
# ==========================================================================

def _get_scoped_stream_slice(
    score: stream.Score,
    part_index: Optional[int] = None,
    measure_start: Optional[int] = None,
    measure_end: Optional[int] = None
) -> Tuple[Optional[stream.Stream], Optional[str]]:
    """
    Get a flattened target stream slice based on optional scoping arguments.
    Return (target_stream, scope_description_string) or (None, error_string).
    """
    target_stream = score
    scope_desc = "the full score"

    if part_index is not None:
        num_parts = len(score.parts)
        if not (0 <= part_index < num_parts):
            return None, (f"Error: Invalid part_index: {part_index}. "
                          f"Score only has {num_parts} parts (indexed 0 to {num_parts - 1}).")
        target_stream = score.parts[part_index]
        scope_desc = f"Part {part_index}"

    if measure_start is not None and measure_end is not None:
        target_stream = target_stream.measures(measure_start, measure_end)
        if part_index is None:
            scope_desc = f"all parts, measures {measure_start}-{measure_end}"
        else:
            scope_desc = f"Part {part_index}, measures {measure_start}-{measure_end}"
            
    flat_target_stream = target_stream.flatten()
         
    # Check the .elements tuple (Score/Part does not have a .hasElements() method)
    if not flat_target_stream.elements: 
         return None, f"Error: The specified scope ({scope_desc}) is empty or contains no musical data."
         
    return flat_target_stream, scope_desc

# ==========================================================================
# RHYTHM TOOL 3: Get Tuplet Statistics (depends on above helper)
# ==========================================================================

def get_tuplet_statistics(
    kern_data: str, 
    part_index: Optional[int] = None,
    measure_start: Optional[int] = None,
    measure_end: Optional[int] = None
) -> str:
    """
    Count occurrences of tuplet groups (e.g., triplets) within the specified scope.
    """
    
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."
        
    target_stream, scope_desc = _get_scoped_stream_slice(
        score, part_index, measure_start, measure_end
    )
    if target_stream is None:
        return scope_desc 

    try:
        tuplet_group_counts = Counter()
        processed_tuplets = set() 

        for n in target_stream.notes:
            if n.duration.tuplets:
                t = n.duration.tuplets[0]
                tuplet_id = id(t)
                
                if tuplet_id not in processed_tuplets:
                    tuplet_type_name = f"{t.numberNotesActual}-in-time-of-{t.durationNormal.type}"
                    
                    if t.numberNotesActual == 3 and t.numberNotesNormal == 2:
                        tuplet_type_name = "Triplet (3-in-2)"
                    elif t.numberNotesActual == 5 and t.numberNotesNormal == 4:
                         tuplet_type_name = "Quintuplet (5-in-4)"
                    elif t.numberNotesActual == 2 and t.numberNotesNormal == 3:
                         tuplet_type_name = "Duplet (2-in-3)"
                    
                    tuplet_group_counts[tuplet_type_name] += 1
                    processed_tuplets.add(tuplet_id)

        if not tuplet_group_counts:
             return f"No tuplet groups were found in {scope_desc}."

        report_parts = [f"Tuplet statistics for {scope_desc}:"]
        for name, count in tuplet_group_counts.most_common():
            report_parts.append(f"- {name}: {count} occurrences")
            
        return " ".join(report_parts)

    except Exception as e:
        return f"Error: An unexpected error occurred during tuplet analysis: {e}"

# ==========================================================================
# EXPRESSION TOOL 1: Get Articulations Statistics (depends on above helper)
# ==========================================================================

def get_articulation_statistics(
    kern_data: str, 
    part_index: Optional[int] = None,
    measure_start: Optional[int] = None,
    measure_end: Optional[int] = None
) -> str:
    """
    Count articulation types (e.g., staccato, accent) within the specified scope.
    """
    
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."
        
    target_stream, scope_desc = _get_scoped_stream_slice(
        score, part_index, measure_start, measure_end
    )
    if target_stream is None:
        return scope_desc

    try:
        articulation_names_found = []
        for event in target_stream.notes: 
            if event.articulations:
                for art in event.articulations:
                    articulation_names_found.append(art.name)

        if not articulation_names_found:
            return f"No articulations were found in {scope_desc}."

        articulation_counts = Counter(articulation_names_found)

        report_parts = [f"Articulation statistics for {scope_desc}:"]
        for name, count in articulation_counts.most_common():
            report_parts.append(f"- {name}: {count} occurrences")
            
        return " ".join(report_parts)

    except Exception as e:
        return f"Error: An unexpected error occurred during articulation analysis: {e}"


# ==========================================================================
# RHYTHM TOOL 4: Get Quarter-Length Sequence for Part & Measure Range
# ==========================================================================

def get_all_durations_in_segment(
    kern_data: str,
    part_index: int,
    measure_start: int,
    measure_end: int
) -> str:
    """
    Extract a per-measure sequence of quarterLength values for a SINGLE specified part
    over a measure range, keeping explicit bar separators ("|") between measures.

    - Includes durations of Notes, Chords, and Rests (all sounding events and silences).
    - Measures are returned in order, each as a comma-separated list of quarterLength values.
    - Measure boundaries are represented by a barline symbol: "|".

    Example return (for measures 1-2):
        "Durations (quarterLength) for Part 0 (Measures 1-2): | 1.0, 0.5, 0.5 | 0.75, 0.25 |"

    Args:
        kern_data (str): Full Humdrum/Kern string of the score
        part_index (int): 0-based index of the part to analyze
        measure_start (int): Starting measure number (inclusive)
        measure_end (int): Ending measure number (inclusive)

    Returns:
        str: Natural language sentence followed by a lead-sheet-like string
             containing per-measure sequences of quarterLength values.
    """

    # 1) Parse
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2) Validate part index
        num_parts = len(score.parts)
        if not (0 <= part_index < num_parts):
            return (f"Error: Invalid part_index: {part_index}. "
                    f"Score only has {num_parts} parts (indexed 0 to {num_parts - 1}).")

        target_part = score.parts[part_index]

        # 3) Slice measure range
        segment = target_part.measures(measure_start, measure_end)
        if not segment:
            return (f"Error: The specified measure range (Part {part_index}, "
                    f"M {measure_start}-{measure_end}) is empty or invalid.")

        # 4) Iterate measures and collect quarterLength values per measure
        per_measure_q_lengths = []
        for m in segment.getElementsByClass('Measure'):
            # Use notesAndRests to include rests as part of rhythmic pattern
            events = list(m.flatten().notesAndRests)
            if not events:
                per_measure_q_lengths.append("")
                continue

            # Format to at most 3 decimals for readability
            def fmt(q):
                try:
                    return (f"{float(q):.3f}").rstrip('0').rstrip('.')
                except Exception:
                    return str(q)

            qls = [fmt(e.duration.quarterLength) for e in events]
            per_measure_q_lengths.append(", ".join(qls))

        final_str = "| " + " | ".join(per_measure_q_lengths) + " |"
        return (f"Durations (quarterLength) for Part {part_index} (Measures {measure_start}-{measure_end}): "
                f"{final_str}")

    except Exception as e:
        return f"Error: An unexpected error occurred while extracting quarter lengths: {e}"


# ==========================================================================
# SCORE STATS TOOL 1: Get Score Structural Statistics
# ==========================================================================

def get_score_structural_statistics(kern_data: str) -> str:
    """
    Global structural statistics: number of parts, measures, total duration, event count, density, ambitus, pitch-class coverage, and average duration.
    """
    
    # 1. Parse the music data
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Error: Parsing failed. Input Kern data may be invalid or empty."

    try:
        # 2. Perform calculations
        
        # Flatten the score once to get note/pitch collections
        flat_score = score.flatten()
        all_notes_and_chords = list(flat_score.notes) # List of Note/Chord objects
        all_pitches = list(flat_score.pitches)      # List of all individual Pitch objects (unpacks chords)

        # --- Basic Counts ---
        part_count = len(score.parts)
        
        # Get highest measure number across all parts
        highest_measure = 0
        for part in score.parts:
            if part.getElementsByClass('Measure'):
                part_highest = part.getElementsByClass('Measure')[-1].number
                highest_measure = max(highest_measure, part_highest)
            
        total_duration_ql = score.highestTime # Most accurate duration in quarter notes
        total_note_events = len(all_notes_and_chords)
        
        # --- Density (handle potential division by zero on empty score) ---
        note_density = 0.0
        if total_duration_ql > 0:
            note_density = total_note_events / total_duration_ql

        # --- Ambitus (Range) (handle empty pitch list) ---
        lowest_note_str = "N/A"
        highest_note_str = "N/A"
        range_semitones = 0
        if all_pitches:
            lowest_pitch = min(all_pitches)
            highest_pitch = max(all_pitches)
            lowest_note_str = lowest_pitch.nameWithOctave
            highest_note_str = highest_pitch.nameWithOctave
            range_semitones = interval.Interval(lowest_pitch, highest_pitch).semitones

        # --- Pitch Variety (Classification Stat) ---
        unique_pc_count = len({p.pitchClass for p in all_pitches})

        # --- Average Duration (Classification Stat) (handle div by zero) ---
        avg_note_duration = 0.0
        if total_note_events > 0:
            total_events_duration_sum = sum([n.duration.quarterLength for n in all_notes_and_chords])
            avg_note_duration = total_events_duration_sum / total_note_events

        # 3. Build the multi-line report string
        # Using '\n' for newlines, which LLM agents can easily parse and display.
        report = [
            "Structural statistics for the score:",
            f"- Total Parts: {part_count}",
            f"- Total Measures (Highest Number): {highest_measure}",
            f"- Total Duration: {total_duration_ql} quarter notes",
            f"- Total Note Events (attacks): {total_note_events}",
            f"- Note Density: {note_density:.2f} events per quarter note",
            "- Pitch Range (Ambitus):",
            f"  - Lowest Note: {lowest_note_str}",
            f"  - Highest Note: {highest_note_str}",
            f"  - Total Span: {range_semitones} semitones",
            f"- Pitch-Class Variety: {unique_pc_count} (out of 12)",
            f"- Average Event Duration: {avg_note_duration:.2f} quarter notes"
        ]
        
        return "\n".join(report)

    except Exception as e:
        return f"Error: An unexpected error occurred during structural analysis: {e}"

def _pick_terminal_chord(chords_in_measure):
    """
    Select the chord for 'phrase-end determination' in the measure:
    Prioritize strong beats (higher beatStrength), then by offset, finally by duration.
    """
    if not chords_in_measure:
        return None
    chords_sorted = sorted(
        chords_in_measure,
        key=lambda c: (c.offset, c.duration.quarterLength, getattr(c, 'beatStrength', 0))
    )
    return chords_sorted[-1]


def detect_cadence_at_measure(kern_data: str, measure_num: int, key_string: str, *, strict_hc_root: bool = False) -> str:
    """
    Determines cadence type (PAC/IAC/HC/Plagal/Deceptive/None) at the 'phrase-end position' 
    (strong beat prioritized) of the specified measure.

    Key optimizations included:
    - Uses format='humdrum' parsing; wraps Part in Score if needed
    - Null protection for measure(...) calls
    - Terminal chord selection prioritizes beatStrength
    - 'Previous chord' prioritizes current measure, falls back to previous measure end
    - Classification logic based on RomanNumeral structural fields (scaleDegree/secondaryRomanNumeral)
    - Provides strict_hc_root control for HC root position requirement (default False)

    Args:
        kern_data (str): Humdrum/Kern format string containing the musical score
        measure_num (int): Measure number to analyze (1-indexed)
        key_string (str): Key signature in standard format (e.g., 'C', 'g#m', 'Bb')
        strict_hc_root (bool): Whether HC requires root position V (default False)

    Returns:
        str: Natural language description of the cadence type:
             - PAC: V(7) → I/i with both chords in root position and tonic in soprano
             - IAC: V(7) → I/i failing one or more PAC conditions
             - HC: Any progression ending on V (optionally root position)
             - Plagal: IV/iv → I/i
             - Deceptive: V(7) → vi/VI
             - None: No clear cadential pattern detected
    """
    # ---------- 1) Parsing and Modeling ----------
    try:
        score = converter.parse(kern_data, format='humdrum')
    except Exception as e:
        return f"Error: Failed to parse Kern data. Details: {e}"

    # If parsing returns a Part, wrap it in a new Score; avoid using explode() to modify content
    if isinstance(score, stream.Part):
        sc = stream.Score()
        sc.insert(0, score)
        score = sc

    if not isinstance(score, stream.Score):
        return "Error: Parsed object is not a Score."

    # ---------- 2) Key ----------
    try:
        key_obj = key.Key(key_string)
    except Exception as e:
        return f"Error: Invalid key string '{key_string}'. Details: {e}"

    # ---------- 3) Chordify and Measure Objects ----------
    chordified_score = score.chordify()

    # Target measure
    m_obj = chordified_score.measure(measure_num)
    if m_obj is None:
        return f"Error: No measure {measure_num} found in the score."

    target_measure_chords = m_obj.getElementsByClass('Chord')
    if not target_measure_chords:
        return f"Error: No chords found in measure {measure_num}."

    # Select terminal chord (strong beat prioritized)
    final_chord = _pick_terminal_chord(target_measure_chords)
    if final_chord is None:
        return f"Error: Could not select a terminal chord in measure {measure_num}."

    # ---------- 4) Finding 'Previous Chord' ----------
    # Prioritize current measure, look back before final_chord (sorted by time and duration)
    chs_curr_sorted = sorted(
        target_measure_chords,
        key=lambda c: (c.offset, c.duration.quarterLength)
    )

    penultimate_chord = None
    if final_chord in chs_curr_sorted:
        idx = chs_curr_sorted.index(final_chord)
        if idx > 0:
            penultimate_chord = chs_curr_sorted[idx - 1]

    # If not found in current measure, look back to previous measure end
    if penultimate_chord is None:
        prev_m_obj = chordified_score.measure(measure_num - 1)
        if prev_m_obj:
            prev_chords = prev_m_obj.getElementsByClass('Chord')
            if prev_chords:
                penultimate_chord = _pick_terminal_chord(prev_chords)

    if penultimate_chord is None:
        return (f"Cadence at M{measure_num} in {key_obj.name}: None "
                f"(Could not find a penultimate chord for analysis)")

    # ---------- 5) Roman Numeral and Structural Fields ----------
    try:
        rn_final = roman.romanNumeralFromChord(final_chord, key_obj)
        rn_prev = roman.romanNumeralFromChord(penultimate_chord, key_obj)
    except Exception:
        return (f"Cadence at M{measure_num} in {key_obj.name}: None "
                f"(Could not determine Roman numerals for the chords)")

    sd_prev = rn_prev.scaleDegree            # 1..7
    sd_curr = rn_final.scaleDegree
    is_applied_prev = (rn_prev.secondaryRomanNumeral is not None)
    is_applied_curr = (rn_final.secondaryRomanNumeral is not None)

    is_V_prev = (sd_prev == 5) and (not is_applied_prev)
    is_I_curr = (sd_curr == 1) and (not is_applied_curr)
    is_IV_prev = (sd_prev == 4) and (not is_applied_prev)
    is_VI_curr = (sd_curr == 6) and (not is_applied_curr)

    # Soprano approximation (highest note from chordify's terminal chord)
    soprano_degree = None
    if final_chord.pitches:
        try:
            soprano_degree = key_obj.getScaleDegreeFromPitch(final_chord.pitches[-1])
        except Exception:
            soprano_degree = None  # Non-scale tones, etc.

    # ---------- 6) Cadence Classification (order matters) ----------
    cadence_type = "None/Uncertain"

    # PAC: V(7)→I(i) with both root position and soprano ^1
    if (is_V_prev and is_I_curr and
        penultimate_chord.inversion() == 0 and
        final_chord.inversion() == 0 and
        soprano_degree == 1):
        cadence_type = "PAC"

    # IAC: V(7)→I(i) but doesn't meet PAC root position/soprano conditions
    elif is_V_prev and is_I_curr:
        cadence_type = "IAC"

    # HC: Ends on V; optionally requires root position
    elif (sd_curr == 5) and (not is_applied_curr):
        if (not strict_hc_root) or (final_chord.inversion() == 0):
            cadence_type = "HC"

    # Plagal: IV→I
    elif is_IV_prev and is_I_curr:
        cadence_type = "Plagal Cadence"

    # Deceptive: V→vi/VI
    elif is_V_prev and is_VI_curr:
        cadence_type = "Deceptive Cadence"

    return (f"Cadence at M{measure_num} in {key_obj.name}: {cadence_type} "
            f"({rn_prev.figure} → {rn_final.figure})")


def get_score_basic_summary(kern_data: str) -> str:
    """
    A simple and robust function to get the number of parts and measures from a kern score.
    This is intended to be a lightweight helper for providing context to an agent.
    """
    score = parse_kern_string_to_score(kern_data)
    if not score:
        return "Score Summary: [Error - Could not parse score]"

    try:
        part_count = len(score.parts)
        highest_measure = 0
        
        # Get the highest measure number across all parts
        for part in score.parts:
            if part.getElementsByClass('Measure'):
                part_highest = part.getElementsByClass('Measure')[-1].number
                highest_measure = max(highest_measure, part_highest)
            
        return f"Score Summary: {part_count} parts, {highest_measure} measures."
    
    except Exception:
        return "Score Summary: [Error - Could not extract structural info]"



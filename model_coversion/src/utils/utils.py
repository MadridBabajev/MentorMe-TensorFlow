def extract_vocabulary_from_transcriptions(vocabulary_file):
    vocab_set = set()
    with open(vocabulary_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # Transcription is everything after the first 9 fields.
            transcription = " ".join(parts[9:]).replace('|', ' ')
            for ch in transcription:
                vocab_set.add(ch)
    return sorted(vocab_set)
def categorize_review(text, aspect_keywords):
    detected_aspects = []
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_aspects.append(aspect)
    if not detected_aspects:
        return ['Umum']
    return detected_aspects
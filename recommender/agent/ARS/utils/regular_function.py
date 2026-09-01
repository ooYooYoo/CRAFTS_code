import re

def _clean_item_text(item):
    if item is None:
        return None

    item = str(item).strip()

    
    item = item.replace("**", "").replace("*", "")
    item = item.replace("`", "")
    item = item.strip().strip('"').strip("'")

    
    item = item.strip(" .,:;，。；：")

    return item.strip()


def split_rec_reponse(response):
    if response is None:
        return None, None

    text = str(response).strip()
    reason = None
    item = None

    
    m = re.search(
        r"Final Recommendation\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )
    if m:
        item = m.group(1).splitlines()[0].strip()

    
    if not item:
        m = re.search(
            r"Item\s*:\s*(.+)",
            text,
            flags=re.IGNORECASE
        )
        if m:
            item = m.group(1).splitlines()[0].strip()

    
    m = re.search(
        r"Reason\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        reason = m.group(1).strip()

    item = _clean_item_text(item)

    return reason, item
# def split_rec_reponse(response):
#     """


#     """

#     if response is None or not isinstance(response, str):
#         print("[split_rec_reponse] invalid response (None or non-string)")
#         return None, None


#     text = response + '\n'
#     pattern = r'Reason: (.*?)\nItem: (.*?)\n'
#     matches = re.findall(pattern, text, re.DOTALL)
#     if len(matches) == 0:

#         pattern_relaxed = r'Reason:\s*(.*?)\s*Item:\s*(.*?)\s*\n'
#         matches = re.findall(pattern_relaxed, text, re.DOTALL)

#     if len(matches) == 0:
#         print("[split_rec_reponse] can not split, response = ", response)
#         return None, None


#     match = matches[0]
#     return match[0].strip(), match[1].strip()


def split_user_response(response):
    
    
    if response is None or not isinstance(response, str):
        print("[split_user_response] invalid response (None or non-string)")
        return None, None

    text = response + '\n'
    pattern = r'Reason: (.*?)\Decision: (.*?)\n'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_reponse] can not split, response = ", response)
        return None, None

    match = matches[0]
    decision_text = match[1].lower().strip()
    if decision_text.startswith('yes'):
        return match[0].strip(), True
    elif decision_text.startswith('no'):
        return match[0].strip(), False
    else:
        print("[split_user_reponse] can not find flag, response = ", response)
        return None, None

def split_user_rec_reponse(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_rec_reponse]can not split, response = ", response)
        return None, None
    match = matches[0]
    return match[0].strip(), match[1].strip()

def split_user_ab_response(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nDecision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_ab_reponse]can not split, response = ", response)
        return None, None

    match = matches[0]
    if match[1].lower().startswith('yes'):
        return match[0].strip(), 1
    elif match[1].lower().startswith('no'):
        return match[0].strip(), 0
    else:
        print("[split_user_ab_reponse]can not find flag, response = ", response)
        return None, None
    
def split_prior_rec_response(response):
    response += '\n'
    pattern = r'Item: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_rec_response]can not split, response = ", response)
        return None
    match = matches[0]
    return match.strip()

def split_prior_llama3_response(response):
    pattern = r'Item: (.*?)<\|eot_id\|>'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_llama3_response]can not split,try split2,  response = ", response)
        return split_prior_rec_response(response)
    match = matches[0]
    return match.strip()

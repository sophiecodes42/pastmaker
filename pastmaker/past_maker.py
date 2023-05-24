import re

import spacy
from pattern.text.en import (
    INDICATIVE,
    PAST,
    PLURAL,
    PROGRESSIVE,
    SINGULAR,
    conjugate,
    lexeme,
)
from spacy.lang.char_classes import (
    ALPHA,
    ALPHA_LOWER,
    ALPHA_UPPER,
    CONCAT_QUOTES,
    LIST_ELLIPSES,
    LIST_ICONS,
)
from spacy.symbols import NOUN
from spacy.tokenizer import Tokenizer
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from spacy.util import compile_infix_regex

# small language model for sentence tokenizing:
nlp_small = spacy.load("en_core_web_sm")
nlp_big = spacy.load("en_core_web_trf")

SUBJ_DEPS = {'agent', 'csubj', 'csubjpass', 'expl', 'nsubj', 'nsubjpass'}
TEMPORAL_REPLACEMENTS_DIC = {"currently": "at the time", "Currently": "At the time", "now": "then", "Now": "Then"} #Todo create constants file and dictionary for replacements

def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights
            if right.dep_ == 'conj']


def is_plural_noun(token):
    """
    Returns True if token is a plural noun, False otherwise.

    Args:
        token (``spacy.Token``): parent document must have POS information

    Returns:
        bool
    """
    if token.doc.is_tagged is False:
        raise ValueError('token is not POS-tagged')
    return True if token.pos == NOUN and token.lemma != token.lower else False


def get_subjects_of_verb(verb):
    if verb.dep_ == ["aux", "auxpass"] and list(verb.ancestors):
        return get_subjects_of_verb(list(verb.ancestors)[0])
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts
             if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    if not len(subjs) or (len(subjs) == 1 and subjs[0].text in ["which", "that", "who"]):
        if len(list(verb.ancestors)):
            return get_subjects_of_verb(list(verb.ancestors)[0])
    return subjs


def is_plural_verb(token):
    if token.doc.is_tagged is False:
        raise ValueError('token is not POS-tagged')
    subjects = get_subjects_of_verb(token)
    if not len(subjects):
        return False
    plural_score = sum([is_plural_noun(x) for x in subjects])/len(subjects)

    return plural_score > 0.0

def preserve_caps(word, newWord):
    """Returns newWord, capitalizing it if word is capitalized."""
    if word[0] >= 'A' and word[0] <= 'Z':
        newWord = newWord.capitalize()
    return newWord

def past_maker(input_text: Doc):
    """Change the tense of text.

    Todo:
    1) when there is a verb coming after a conjunction (more than one verb belonging to a subject/infinitive), every verb should have the same "change" applied to:
        e.g. "to evaluate ...and characterize" and NOT "to evaluate ...and characterized"
    2) ignore adverbs inbetween subject and verb or modal and verb:
        e.g."a concentration of...mL will only be used" ---> "...mL were only used" and NOT "...mL will only was used"
    3) incorrect past passive forms have to be fixed:
        e.g. "medication will be given" --> "medication was given" and NOT "medication was gave"

    [..]

    :param input_text: the original sentence (processed by spacy 'en_core_web_trf' language model) that should be transformed into past tense
    :return text_out: The whole input_text-sentence transformed with all changes made within the function
    :return changes: lists all changes made in the input text with change[0] being the original string and change[1] the transformed string
    """
    # Todo: Question: should it stay this way, that words get changed token for token or is there a way to change whole "tense" of a sentence?

    tense = PAST  # this is only used for past making

    if not len(input_text):
        return "", []

    out = list()
    changes = []
    pattern_stopiteration_workaround()
    for sent in nlp_small(input_text).sents:
        sent = nlp_big(sent.text)
        # print([(word.text, word.tag_, word.morph) for word in sent])
        out.append(sent[0].text)
        words = []
        head_conjuncts = {"head": None, "conjunct_verbs": []}

        for word in sent:
            aux_construction = False
            # if word is verb, look for head of conjuncts:
            if word.tag_ in ("VB", "VBD", "VBP", "VBZ", "VBN", "MD"):
                if word.head == word:
                    head_conjuncts.update({"head": word, "conjunct_verbs": _get_conjuncts(word)})
            words.append(word)

            if len(words) == 1:
                continue
            if (
                word.text.capitalize() == "See"
                or (
                    re.search("VerbForm=(Inf|Ger)", str(word.morph))
                    and not (
                        is_in_modal_list(words[-2])
                        or (len(words) > 2 and is_in_modal_list(words[-3]) and is_adverb_or_NOT(words[-2]))
                    )
                )
                or words[-2].text in "/\\"
                or any(
                    string_y in words[-1].text for string_y in [":", "<", ">", "≥", "≤", "=", "/", "°", "´", "`", "AEs"]
                )
                or words[-1].text.isupper()
            ):  # or (re.search("VerbForm=(Part|Inf|Ger)", str(word.morph)) and not str(word.morph) == "AUX"):
                out.append(words[-1].text)
                continue

            # change future tense (line 215 & line 217) and present tense (line 216) to past tense
            if (
                (is_will(words[-2]) and words[-2].tag_ == "MD" and words[-1].tag_ == "VB")
                or (
                    not words[-2].text.lower() == "to"
                    and words[-1].tag_ in ["VB", "VBD", "VBP", "VBZ", "VBN"]
                    and not words[-1].text.isupper()
                )
                or (
                    len(words) > 2
                    and is_will(words[-3])
                    and words[-3].tag_ == "MD"
                    and words[-1].tag_ == "VB"
                    and is_adverb_or_NOT(words[-2])
                )
            ):
                this_tense = PAST
                subjects = [x.text.lower() for x in get_subjects_of_verb(words[-1])]

                if any(elem in subjects for elem in ["i", "we"]):
                    person = 1
                elif any(elem == "you" for elem in subjects):
                    person = 2
                else:
                    person = 3
                # to find out whether it's were/was or has/have
                number = PLURAL if is_plural_verb(words[-1]) else SINGULAR

                # find verb phrases with auxiliaries to check which verb to change and where to put adverbs
                if (is_will(words[-2]) and words[-2].tag_ == "MD") or words[-2].text == "had":
                    make_change(changes, out, "pop", -1)
                    aux_construction = True
                elif (
                    len(words) > 2
                    and is_adverb_or_NOT(words[-2])
                    and ((is_will(words[-3]) and words[-3].tag_ == "MD") or words[-3].text == "had")
                ):
                    make_change(changes, out, "pop", -2)
                    aux_construction = True

                # is/are presented/recommended stays as is; will be presented -> is/are presented
                if words[-1].tag_ == "VBN":
                    out.append(words[-1].text)
                    if words[-1].text in ["presented", "recommended"]:
                        changes, out = do_not_change_reference(words, changes, out)

                # Exceptions to the rules: should/must be -> had to be
                elif words[-1].tag_ in ["VB", "VBP", "VBZ", "MD"]:
                    if words[-2].text.lower() in ["should", "must"]:
                        make_change(changes, out, "pop", -1)
                        conjugated_verb = "had to " + words[-1].text
                    elif (
                        len(words) > 2 and is_adverb_or_NOT(words[-2]) and words[-3].text.lower() in ["should", "must"]
                    ):
                        make_change(changes, out, "pop", -2)
                        make_change(changes, out, "pop", -1)
                        conjugated_verb = "had to " + words[-2].text + " " + words[-1].text

                    ## would is removed and verb transformed into past tense 'would do' --> 'did'
                    elif words[-2].text.lower() == "would" and words[-2].tag_ == "MD":
                        make_change(changes, out, "pop", -1)
                        conjugated_verb = conjugate(words[-1].text, tense=this_tense, person=person, number=number)

                    # can/might/may (+ not) + verb = could (+ not) + verb
                    elif is_canmightmay(words[-2]):
                        make_change(changes, out, "replace", -1, new_val="could")
                        conjugated_verb = words[-1].text
                    elif len(words) > 2 and is_canmightmay(words[-3]) and is_adverb_or_NOT(words[-2]):
                        make_change(changes, out, "replace", -2, new_val="could")
                        conjugated_verb = words[-1].text
                    elif words[-2].text.lower() == "cannot":
                        make_change(changes, out, "replace", -1, new_val="could not")
                        conjugated_verb = words[-1].text

                    ## past transformation from are/is:
                    elif words[-1].text.lower() == "are":
                        conjugated_verb = "were"
                    elif words[-1].text.lower() == "is":
                        conjugated_verb = "was"

                    else:
                        conjugated_verb = conjugate(words[-1].text, tense=this_tense, person=person, number=number)
                    ## "not" changes position from before main verb to after main verb: will not --be-- --> --was-- not
                    if not (is_adverb_or_NOT(words[-2]) and aux_construction):
                        make_change(changes, out, "append_replaced", old_val=words[-1].text, new_val=conjugated_verb)
                    else:
                        if words[-1].text.lower() == "be":
                            make_change(
                                changes,
                                out,
                                "switch_last_and_new",
                                -1,
                                old_val=words[-1].text,
                                new_val=f"{conjugated_verb} {words[-2].text}",
                            )
                        else:
                            new_val = (
                                f"did {words[-2].text} {words[-1].text}"
                                if words[-2].text.lower() == "not"
                                else f"{words[-2].text} {conjugated_verb}"
                            )
                            make_change(changes, out, "pop", -1)
                            make_change(
                                changes,
                                out,
                                "append_replaced",
                                old_val=words[-2].text,
                                new_val=new_val,
                            )
                # fix for verbs in past tense "was" --> had been
                elif words[-1].tag_ == "VBD":
                    conjugated_verb = conjugate(
                        words[-1].text, tense=this_tense, mood=INDICATIVE, aspect=PROGRESSIVE, alias="ppart", tag="VBN"
                    )
                    make_change(
                        changes, out, "append_replaced", old_val=words[-1].text, new_val="had " + conjugated_verb
                    )

                # TODO: verbs at the beginning of the sentence (e.g. bulletlist in inclusion/exclusion criteria "Has regular menstrual cycle...")
            else:
                out.append(words[-1].text)

            # negation
            if (
                len(words) > 2
                and words[-2].text.lower() + words[-1].text.lower() in ("didnot", "donot", "wouldnot")
                and not re.search("VerbForm=Inf", str(words[-2].morph))
            ):
                if tense == PAST:
                    make_change(changes, out, "replace", -2, new_val="did")
                else:
                    make_change(changes, out, "pop", -2)

    text_out = " ".join(out)

    # #   Writing tense Changes to txt file for debugging purpose!
    # if changes:
    #     for change in changes:
    #         print(change[0] + " => " + change[1] + "\n")

    return text_out + " ", changes

def is_adverb_or_NOT(token: Token) -> bool:
    """
    checks whether a token is an adverb or (e.g. in case of cannot) also checks for 'not'
    """
    return token.tag_ in ["RB", "RBR", "RBS"] or token.text.lower() == "not"


def is_canmightmay(token: Token) -> bool:
    """
    checks whether a token is either can, might or may
    """
    return token.text.lower() in ["can", "might", "may"]


def do_not_change_reference(
    words: list[Token], changes: list[list[str]], out: list[str]
) -> tuple[list[list[str]], list[str]]:
    """
    does not transform e.g. "is/are presented in Section X.X.X" into past tense
    """
    if words[-2].text == "is":
        make_change(changes, out, "replace", -2, new_val="is")
    elif words[-2].text == "are":
        make_change(changes, out, "replace", -2, new_val="are")
    if len(words) > 2:
        if words[-1].text == "presented" and words[-2].text == "be" and words[-3].text == "will":
            make_change(changes, out, "replace", -2, new_val="are") if is_plural_verb(words[-3]) else make_change(
                changes, out, "replace", -2, new_val="is"
            )
    return changes, out


def is_in_modal_list(token: Token) -> bool:
    """
    checks whether token is a certain modal verb
    """
    return token.text.lower() in ["must", "should", "will", "can", "might", "may", "cannot"]


def is_will(token: Token) -> bool:
    """
    checks whether token is 'will'
    """
    return token.text.lower() == "will"


def pattern_stopiteration_workaround():
    """
    This function is just a dummy workaround for the pattern-package which can have an issue the first time it is called,
    so it needs to be called once first without having an impact on the output.
    """
    try:
        lexeme("gave")
    except Exception:
        pass

def make_change(
    changes_array: list[list[str]],
    output_array: list[str],
    change_type: str,
    index: int = 0,
    old_val: str = "",
    new_val: str = "",
):
    """
    This function uses the changes saved in the changes_arr (list of [original_str, transformed_str]-pairs)
    and performs action defined in "change_type" in case the transformation has to be adapted after initial past transformation of a verb
    e.g. due to dependency parsing rules (verbs are dependent on each other and cannot be observed individually)

    :param changes_array: (list of [original_str, transformed_str]-pairs)
    :ptype changes_array: list of lists containing str-pairs
    :param output_array: list of strings which are added subsequently after making changes to the tokens
    :ptype output_array: list
    :param change_type: action to be performed to the planned changes (sometimes a change is made to a verb but hast to be adapted due to following dependent verbs)
    :ptype change_type: str
    :param index: position of element within the array (changes and output) that needs to be changed
    :ptype index: int
    :param old_val: part of input_text that is considered to be transformed with one of the change_types
    :ptype old_val: str
    :param new_val: string that should appear instead of old_val if this change is implemented
    :ptype new_val: str

    """

    if change_type == "pop":
        changes_array.append([output_array[index], ""])
        output_array.pop(index)

    elif change_type == "pop_last":  # index = -1
        if not len(changes_array):
            return
        changes_array[index][1] = ""
        output_array.pop(index)

    elif change_type == "append":
        changes_array.append(["", new_val])
        output_array.append(new_val)

    elif change_type == "replace":
        changes_array.append([output_array[index], new_val])
        output_array[index] = new_val

    elif change_type == "append_replaced":
        changes_array.append([old_val, new_val])
        output_array.append(new_val)

    elif change_type == "switch_last_and_new":  # index = -1
        if len(changes_array):
            changes_array[index][1] = ""
            output_array.pop(index)
        changes_array.append([old_val, new_val])
        output_array.append(new_val)

    if changes_array[-1][0] == changes_array[-1][1]:  # no change
        changes_array.pop()

def custom_tokenizer(nlp):
    """
    This function is retrieved from https://stackoverflow.com/questions/58105967/spacy-tokenization-of-hyphenated-words
    It modifies the tokenizer of a spacy language model, so that words with hyphens (e.g. off-white) are seen as one token
    """

    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>≥≤=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,
        rules=nlp.Defaults.tokenizer_exceptions,
    )
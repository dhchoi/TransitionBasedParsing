# http://ilk.uvt.nl/conll/#dataformat
# ID  FORM  LEMMA  CPOSTAG  POSTAG  FEATS  HEAD  DEPREL  PHEAD  PDEPREL
# 0   1     2      3        4       5      6     7       8      9

# ID: Token counter, starting at 1 for each new sentence.
ID = 0

# FORM: Word form or punctuation symbol.
FORM = 1

# LEMMA: Lemma or stem (depending on particular data set) of word form, or an underscore if not available.
LEMMA = 2

# CPOSTAG: Coarse-grained part-of-speech tag, where tagset depends on the language.
CPOSTAG = 3

# POSTAG: Fine-grained part-of-speech tag, where the tagset depends on the language,
#         or identical to the coarse-grained part-of-speech tag if not available.
POSTAG = 4

# FEATS: Unordered set of syntactic and/or morphological features (depending on the particular language),
#        separated by a vertical bar (|), or an underscore if not available.
FEATS = 5

# HEAD: Head of the current token, which is either a value of ID or zero ('0').
HEAD = 6

# DEPREL: Dependency relation to the HEAD.
DEPREL = 7

# PHEAD: Projective head of current token, which is either a value of ID or zero ('0'), or underscore if not available.
PHEAD = 8

# PDEPREL: Dependency relation to the PHEAD, or an underscore if not available.
PDEPREL = 9

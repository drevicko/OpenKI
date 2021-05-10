#   Copyright (c) 2020.  Dongxu Zhang.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
from typing import Optional


def extract_window(sentence:str, sname:str, oname:str)->Optional[str]:
    """
    Return a heuristic predicate string for the relation between sname and oname expressed in sentences. The heuristic
    is to start 3 words before the subject, mask the subject/object name with <subj>/<obj> and end 4 words after the
    object. If the subject or the object aren't found, returns None. This function was provided by Dongxu, 1st author
    of the OpenKI paper.
    :param sentence:
    :param sname: name of subject
    :param oname: name of object
    :return: masked window around subject to object word span or None
    """
    sent = ' '.join(sentence.split())
    s_index = sent.find(sname)
    if s_index == -1:
        return None
    sent = sent.split()
    sid = 0# save the position of the first word of subject
    count = 0
    while count < s_index:
        count += len(sent[sid]) + 1
        sid += 1
    sent[sid:sid+len(sname.split())] = ['<subj>']

    sent = ' '.join(sent)
    o_index = sent.find(oname)
    if o_index == -1:
        return None
    sent = sent.split()
    oid = 0# save the position of the first word of object
    count = 0
    while count < o_index:
        count += len(sent[oid]) + 1
        oid += 1
    sent[oid:oid+len(oname.split())] = ['<obj>']

    try:
        sid = sent.index('<subj>')
        oid = sent.index('<obj>')
    except:
        return None
    if sid == oid:
        return None
    elif sid < oid:
        sid = sid - 3 if sid - 3 >= 0 else 0
        oid = oid + 4 if oid + 4 <= len(sent) else len(sent)
        return ' '.join(sent[sid:oid])
    else:
        oid = oid - 3 if oid - 3 >= 0 else 0
        sid = sid + 4 if sid + 4 <= len(sent) else len(sent)
        return ' '.join(sent[oid:sid])



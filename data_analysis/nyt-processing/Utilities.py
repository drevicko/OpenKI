#   Copyright (c) 2020.  CSIRO Australia.
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
# from src.python import Document_pb2
import Document_pb2
from google.protobuf.internal.decoder import _DecodeVarint32


def message_iterator(input_stream, message_name="Relation", reuse_msg_object=False):
    if reuse_msg_object:
        message = getattr(Document_pb2, message_name)()
    buf = input_stream.read()
    next_pos, pos = 0, 0
    while pos < len(buf):
        if not reuse_msg_object:
            message = getattr(Document_pb2, message_name)()
        length, bytes_read = _DecodeVarint32(buf[pos:], 0)
        pos += bytes_read
        bytes_read = message.ParseFromString(buf[pos: pos + length])
        if bytes_read != length:
            print(f"{bytes_read} != {length}; {type(bytes_read)}, {type(length)}")
            exit(0)
        pos += bytes_read
        yield message


def base32(num, numerals="0123456789bcdfghjklmnpqrstvwxyz_"):
    '''ref: http://code.activestate.com/recipes/65212/'''
    return ((num == 0) and numerals[0]) or (base32(num // 32, numerals).lstrip(numerals[0]) + numerals[num % 32])


def guid_to_mid(guid):
    '''
    Convert guid to mid.
    Conversion rules mentioned at http://wiki.freebase.com/wiki/Guid
    '''
    try:
        guid_ = guid.split('/')[-1]
        # remove 9202a8c04000641f8
        guid_ = guid_[17:]
        i = int(guid_, 16)
        mid = 'm.0' + str(base32(i))
        return mid
    except ValueError:
        return guid

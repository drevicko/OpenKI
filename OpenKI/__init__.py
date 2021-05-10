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

# from OpenKI.OpenKI_Data import *
# from OpenKI.RelationScorers import *
# from OpenKI.LossFunctions import *
# from OpenKI.Constants import *
# from OpenKI.Evaluation import *
# from OpenKI.UtilityFunctions import *
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging_handler = logging.StreamHandler(sys.stdout)
logging_handler.setLevel(logging.INFO)

logging_formatter = logging.Formatter(fmt='%(levelname).1s %(asctime)s [%(filename)s:%(lineno)s] %(message)s',
                                      datefmt='%m-%d %H:%M:%S')
logging_handler.setFormatter(logging_formatter)
logger.addHandler(logging_handler)

# workaround for double logging issue from absl introduced by tensorboard
logger.propagate = False

# # NOTE: alternative workaround for absl ouble logging issue:
# import absl.logging
# logging.root.removeHandler(absl.logging._absl_handler)
# absl.logging._warn_preinit_stderr = False

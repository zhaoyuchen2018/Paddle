# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def check_import_scipy(OsName):
    print_info = ""
    if OsName == 'nt':
        try:
            import scipy.io as scio
        except ImportError as e:
            print_info = str(e)
        if (len(print_info) > 0):
            if 'DLL load failed' in print_info:
                raise ImportError(
                    print_info +
                    "\nplease download visual C++ Redistributable for vs 2015, https://www.microsoft.com/en-us/download/details.aspx?id=48145"
                )
    return

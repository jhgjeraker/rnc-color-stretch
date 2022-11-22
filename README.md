# rnc-color-stretch
Python adaptation of `rnc-color-stretch`, an automatic image stretching tool used for astrophotography post-processing originally written in [davinci](http://davinci.asu.edu/index.php?title=Main_Page) by [Roger N. Clark](https://clarkvision.com/articles/astrophotography.software/rnc-color-stretch/).

## Motivation
My goal was to make this interesting piece of software more available through Python. It was also a nice excercise for me to learn about programmatical image processing. You can read more at https://gjeraker.com/content/projects/rnc-color-stretch.html.

## Usage
 All code is contained in a single file and can be run by the following commands.

```
python rnc_color_stretch.py <image>.tiff
```

Configuration parameters can be seen by calling with the helper flag.

```
python rnc_color_stretch.py --help
```

## Copyright and Licence

Copyright (c) 2016, Roger N. Clark, clarkvision.com

http://www.clarkvision.com/articles/astrophotography.software/rnc-color-stretch/

All rights reserved.

GNU General Public License https://www.gnu.org/licenses/gpl.html

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
- Redistributions of the program must retain the above copyright notice, this list of conditions and the following disclaimer.
- Neither Roger N. Clark, clarkvision.com nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

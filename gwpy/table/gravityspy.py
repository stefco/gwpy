# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Extend :mod:`astropy.table` with the `GravitySpyTable`
"""

import os

from ..utils import mp as mp_utils
from .table import EventTable
import numpy as np

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['GravitySpyTable']


class GravitySpyTable(EventTable):
    """A container for a table of Gravity Spy Events (as well as
    Events from the O1 Glitch Classification Paper whcih includes

    - PCAT
    - PC-LIB
    - WDF
    - WDNN
    - Karoo GP

    This differs from the basic `~astropy.table.Table` in two ways

    - GW-specific file formats are registered to use with
      `GravitySpyTable.fetch`
    - columns of this table are of the `EventColumn` type, which provides
      methods for filtering based on a `~gwpy.segments.SegmentList` (not
      specifically time segments)

    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `GravitySpyTable`
    """

    # -- i/o ------------------------------------

    def download(self, **kwargs):
        """If table contains Gravity Spy triggers `EventTable`

        Parameters
        ----------
        nproc : `int`, optional, default: 1
            number of CPUs to use for parallel file reading

        download_path : `str` optional, default: 'download'
            Specify where the images end up.

        download_durs : `list` optional, default: [0.5, 1.0, 2.0, 4.0]
            Specify exactly which durations you want to download
            default is to download all the avaialble GSpy durations.

        kwargs: Optional TrainingSet and LabelledSamples args
            that will download images in a special way
            ./"Label"/"SampleType"/"image"

        Returns
        -------
        Folder containing omega scans sorted by label
        """
        from six.moves.urllib.request import urlopen
        import os
        # back to pandas
        try:
            imagesDB = self.to_pandas()
        except ImportError as exc:
            exc.args = ('pandas is required to download triggers',)
            raise

        # Remove any broken links
        imagesDB = imagesDB.loc[imagesDB.imgUrl1 != '?']

        TrainingSet = kwargs.pop('TrainingSet', 0)
        LabelledSamples = kwargs.pop('LabelledSamples', 0)
        download_location = kwargs.pop('download_path',
                                       os.path.join('download'))
        duration_values = np.array([0.5, 1.0, 2.0, 4.0])
        download_durs = kwargs.pop('download_durs', duration_values)

        duration_idx = []
        for idur in download_durs:
            duration_idx.append(np.argwhere(duration_values == idur)[0][0])

        duration_values = duration_values[duration_idx]
        duration_values = np.array([duration_values]).astype(str)

        # LabelledSamples are only available when requesting the
        if LabelledSamples:
            if 'SampleType' not in imagesDB.columns:
                raise ValueError('You have requested Labelled Samples '
                                 'for a Table which does not have '
                                 'this column. Did you fetch a '
                                 'trainingset* table?')

        # If someone wants labelled samples they are
        # Definitely asking for the training set but
        # may hve forgotten
        if LabelledSamples and not TrainingSet:
            TrainingSet = 1

        # Let us check what columns are needed
        cols_for_download = ['imgUrl1', 'imgUrl2', 'imgUrl3', 'imgUrl4']
        cols_for_download = [cols_for_download[idx] for idx in duration_idx]
        cols_for_download_ext = ['Label', 'SampleType', 'ifo', 'uniqueID']

        if not TrainingSet:
            imagesDB['Label'] = ''
        if not LabelledSamples:
            imagesDB['SampleType'] = ''

        if not os.path.isdir(download_location):
            os.makedirs(download_location)

        if TrainingSet:
            for iLabel in imagesDB.Label.unique():
                if LabelledSamples:
                    for iType in imagesDB.SampleType.unique():
                        if not os.path.isdir(os.path.join(
                                             download_location,
                                             iLabel, iType)):
                            os.makedirs(os.path.join(download_location,
                                        iLabel, iType))
                else:
                    if not os.path.isdir(os.path.join(download_location,
                                                      iLabel)):
                        os.makedirs(os.path.join(download_location,
                                                 iLabel))

        images_for_download = imagesDB[cols_for_download]
        images = images_for_download.as_matrix().flatten()
        images_for_download_ext = imagesDB[cols_for_download_ext]
        duration = np.atleast_2d(
                                 duration_values.repeat(
                                   len(images_for_download_ext), 0).flatten(
                                     )).T
        images_for_download_ext = images_for_download_ext.as_matrix(
                                       ).repeat(len(cols_for_download), 0)
        images_for_for_download_path = np.array([[download_location]]).repeat(
                                       len(images_for_download_ext), 0)
        images = np.hstack((np.atleast_2d(images).T,
                           images_for_download_ext, duration,
                           images_for_for_download_path))

        def get_image(url):
            name = url[3] + '_' + url[4] + '_spectrogram_' + url[5] + '.png'
            outfile = os.path.join(url[6], url[1], url[2], name)
            with open(outfile, 'wb') as fout:
                fout.write(urlopen(url[0]).read())

        # calculate maximum number of processes
        nproc = min(kwargs.pop('nproc', 1), len(images))

        # define multiprocessing method
        def _download_single_image(url):
            try:
                return url, get_image(url)
            except Exception as exc:  # pylint: disable=broad-except
                if nproc == 1:
                    raise
                else:
                    return url, exc

        # read files
        output = mp_utils.multiprocess_with_queues(
            nproc, _download_single_image, images)

        # raise exceptions (from multiprocessing, single process raises inline)
        for f, x in output:
            if isinstance(x, Exception):
                x.args = ('Failed to read %s: %s' % (f, str(x)),)
                raise x

    @classmethod
    def search(cls, uniqueID, howmany=10):
        """perform restful API version of search available here:
        https://gravityspytools.ciera.northwestern.edu/search/

        Parameters
        ----------
        uniqueID : `str`,
            This is the unique 10 character hash that identifies
            a Gravity Spy Image

        howmany : `int`, optional, default: 10
            number of similar images you would like

        Returns
        -------
        `GravitySpyTable` containing similar events based on
        an evaluation of the Euclidean distance of the input image
        to all other images in some Feature Space
        """
        from astropy.utils.data import get_readable_fileobj
        import json
        from six.moves.urllib.error import HTTPError

        # Need to build the url call for the restful API
        base = 'https://gravityspytools.ciera.northwestern.edu' + \
            '/search/similarity_search_restful_API'

        parts = {
            'howmany': howmany,
            'imageid': uniqueID,
        }

        search = '&'.join('{}={}'.format(key, value) for
                          key, value in parts.items())

        url = '{}/?{}'.format(base, search)

        try:
            with get_readable_fileobj(url) as f:
                return GravitySpyTable(json.load(f))
        except HTTPError as exc:
            if exc.code == 500:
                exc.msg = exc.msg + ', please confirm the uniqueID is valid'
                raise

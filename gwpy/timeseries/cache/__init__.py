# -*- coding: utf-8 -*-
# Copyright (C) Stefan Countryman (2018)
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

"""
TimeSeries Caching

This module provides functions and constants used to cache gravitational wave
timeseries data and provides the following extended base TimeSeries classes
with inbuilt caching functionality to be used as base classes for other
TimeSeries-related classes.

-------------------------   ---------------------------------------------------
`CacheableTimeSeries`       extension of the `TimeSeriesBase` class providing
                            caching functionality for all data-retrieval
                            methods to improve speed and reliability on
                            repeated, long-running, and/or failure-prone
                            queries.
`CacheableTimeSeriesDict`   same as above, but extends the `TimeSeriesBaseDict`
                            class with caching functionality.
-------------------------   ---------------------------------------------------

It also provides functions for working with cached data directly, though the
average user will probably not find these useful.

**Neither of these objects are really designed to be used other than as bases
for user-facing objects.**
"""

from __future__ import print_function

import os
from os import makedirs
import warnings
from threading import Timer
from subprocess import Popen, PIPE
import numbers
from collections import namedtuple
from glob import glob
try:
    from urllib.parse import quote_plus as quote  # , unquote_plus as unquote
except ImportError:
    from urllib import quote_plus as quote  # , unquote_plus as unquote  # py 2

import numpy as np
from numpy import nan
from astropy.units import Quantity

from ...time import (to_gps)
from ...detector import (Channel)
from ..core import (TimeSeriesBase, TimeSeriesBaseDict)
from ..io.losc import (_get_default_caching_behavior)

DEFAULT_CACHEDIR = os.path.expanduser("~/.local/share/gwpy/cache")
DEFAULT_MISSING_VALUE = nan
# TODO Optimize the minimum query length with some testing; 1s is just a sane
# lower-bound, but it will probably be closer to 64s (smallest common frame
# duration).
MIN_QUERY_SECONDS = 1
# make assumptions about sample rates
MAX_SAMPLE_RATE = 2**16
MIN_DT = 1./MAX_SAMPLE_RATE
TREND_TYPES = ("m-trend", "s-trend")
_DIRNAME_SUFFIX = "-Hz"  # suffix for raw channel cache directories
NDS_QUERY_CHANNEL_NUMBER_PREFIX = 'Number of channels received = '
TREND_STATISTICS = ['mean', 'min', 'max', 'rms', 'n']
SUBDIR_DURATION = 1e5

_DictQueryTuple = namedtuple('_DictQueryTuple', ('channels', 'start', 'end'))


class DictQuery(_DictQueryTuple):
    """A tuple of the form (channels, start, end) matching the function
    signature of `TimeSeriesBaseDict` query functions like `fetch` and `find`,
    i.e. `channels` is a list of channels and `start` and `end` are start an
    end times of the queries. Can be passed as `*args` to those fetching
    methods. Provides descriptive property names for the contained data as well
    as a definition of addition for two `DictQuery` instances that returns
    single query tuple containing a superset of the data described by the two
    input queries taken together.

    Parameters
    ----------
    channels : array-like
        a list of channels understood by GWpy.

    start
        a start time for the query in a format understood by GWpy (e.g. a GPS
        time).

    end
        an end time for the query in a format understood by GWpy (e.g. a GPS
        time).

    See Also
    --------
    TimeSeriesBaseDict.fetch
        a method for fetching a `TimeSeriesBaseDict` with the requested
        channels and start/stop times *from a remote NDS2 server*.

    TimeSeriesBaseDict.find
        a method for fetching a `TimeSeriesBaseDict` with the requested
        channels and start/stop times *from locally-saved GW frame files*.

    TimeSeriesBaseDict.get
        a method for fetching a `TimeSeriesBaseDict` with the requested
        channels and start/stop times *from whichever data source is most
        readily available*.
    """

    def __new__(cls, channels, start, end):
        if start > end:
            raise ValueError("start must come before end.")
        return _DictQueryTuple.__new__(cls, sorted(channels), start, end)

    def __add__(self, other):
        return type(self)(
            sorted(set(self.channels).union(set(other.channels))),
            min(self.start, other.start),
            max(self.end, other.end)
        )


_NDSQueryInfoTuple = namedtuple("_NDSQueryInfoTuple",
                                ('remote_channel', 'sample_rate',
                                 'channel_type'))


class NDSQueryInfo(_NDSQueryInfoTuple):
    """Holds channel information from NDS Queries. Explicitly casts
    `sample_rate` to a `float`. A `namedtuple` subclass with properties:

    remote_channel : `str`
        the name of the channel as found on the remote server

    sample_rate : `float`
        the sample rate of the channel found on the remote server

    channel_type : `str`
        a description of the channel type. "raw" for raw channels, "s-trend"
        for second trend, and "m-trend" for minute trend data.
    """

    def __new__(cls, remote_channel, rate, channel_type):
        return _NDSQueryInfoTuple.__new__(cls, remote_channel, float(rate),
                                          channel_type)


def flip(dic):
    """Map the values of `dic` to lists of the keys of `dic`.

    Parameters
    ----------
    dic : `dict`, array-like
        a `dict` or something that can be cast to a `dict`.

    Returns
    -------
    flipped : `dict`
        a dict whose keys are values in `dic` and whose values are keys from
        `dic`.

    Examples
    --------
    >>> flip({'x': 1, 'y': 2, 'z': 1})
    {1: ['x', 'z'], 2: ['y']}
    """
    if not isinstance(dic, dict):
        dic = dict(dic)
    return dict((v, [k for k in dic if dic[k] == v]) for v in
                set(dic.values()))


def nds_query_list(query, host, port, timeout=None):
    """Run an `nds_query -l <query>` command to find NDS2 channels matching the
    given `query`.

    Parameters
    ----------
    query : `str`, `~gwpy.detector.Channel`
        the name (not necessarily complete) to read, or a `Channel` object
        containing the desired query as its channel name.

    host : `str`
        URL of NDS server to use.

    port : `int`
        port number for NDS server query.

    timeout : `float`, `int`, optional
        if provided, specify the number of seconds that should pass before an
        NDS query is cancelled. This is the time *per query*, not the overall
        run time of the function.

    Returns
    -------
    matches : `list`
        a list of `NDSQueryInfo` instances containing information on matching
        channels discovered.

    Raises
    ------
    IOError
        if `nds_query` encountered an error, timed out, or returned unexpected
        output (i.e. if anything went wrong with the request).

    See Also
    --------
    NDSQueryInfo
        the class of the returned channel information.
    """
    cmd = ['nds_query', '-n', host, '-p', str(port), '-l', str(query)]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    if timeout is not None:
        timer = Timer(timeout, lambda p: p.kill(), [proc])
        timer.start()
    out, err = proc.communicate()
    # if we got some kind of error, warn the user and keep going
    if proc.returncode:
        msg = "query query failed.\nCMD: {}\nSTDOUT:\n{}\nSTDERR:\n{}"
        raise IOError(msg.format(cmd, out, err))
    # decode the response and split it into lines
    response = out.decode("utf-8").strip().split('\n')
    # if response doesn't match the expected format, complain and move on
    if not response[0].startswith(NDS_QUERY_CHANNEL_NUMBER_PREFIX):
        msg = "unexpected NDS2 server response: {}".format(response)
        raise IOError(msg)
    # the third line onward should be matching channels
    return [NDSQueryInfo(*line.split()[:3]) for line in response[2:]]


def nds_query_info(channel, epoch=None, host=None, port=None, timeout=None):
    """Query NDS2 servers using the command line `nds_query` client to get
    channel metadata.

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, optional
        GPS epoch of data requested, e.g. the start time of a query.

    host : `str`, optional
        URL of NDS server to use, if blank will try any server
        (in a relatively sensible order) to get the data. Will be ignored
        unless you provide `port` as well.

    port : `int`, optional
        port number for NDS server query, must be given with `host`. Will be
        ignored unless you provide `host` as well.

    timeout : `float`, `int`, optional
        if provided, specify the number of seconds that should pass before an
        NDS query is cancelled. This is the time *per query*, not the overall
        run time of the function.

    Returns
    -------
    channel_info : `NDSQueryInfo`
        remote channel name, smple rate, and trend type of the matching channel
        result.

    Raises
    ------
    IOError
        if the requested channel information cannot be found unambiguously.

    See Also
    --------
    NDSQueryInfo
        the class of the returned channel information.
    """
    from ...io import nds2 as io_nds2
    basename, stat, trend = splitchan(channel)
    # querying nds2 requires that we not include the trend name in the query.
    if stat is not None:
        channel = "{}.{}".format(basename, stat)
    channel = Channel(channel)
    if host is not None and port is not None:
        hosts = [(host, port)]
    else:
        if epoch is None:
            hosts = io_nds2.host_resolution_order(channel.ifo)
        else:
            hosts = io_nds2.host_resolution_order(channel.ifo, epoch=epoch)
    for server, port_ in hosts:
        try:
            matches = nds_query_list(channel, server, port_, timeout)
        except IOError as err:
            warnings.warn(str(err))
            continue
        if len(matches) == 0:  # pylint: disable=len-as-condition
            continue
        # if we are looking at m-trend or s-trend data, we will have multiple
        # responses (since nds_query does not distinguish between the two), and
        # we must therefore downselect our matches to match our trend type.
        if trend is not None:
            matches = [m for m in matches if trend in m.remote_channel]
        if len(matches) != 1:
            warnings.warn(
                ("Found {} results: {}\nsuggesting channel {} is not an "
                 "actual channel!").format(len(matches), matches, channel)
            )
            continue
        return matches[0]
    raise IOError("Could not find requested channel information.")


def _dirname_from_rate(rate):
    """Take a Quantity `rate` and get back a descriptive directory name."""
    return str(float(rate.to("1/s").value)) + _DIRNAME_SUFFIX


def _matching_raw_dirs(channel, cachedir=DEFAULT_CACHEDIR):
    """Get a list of full paths of directories that might contain raw data for
    `channel`. There should be at most one such directory, so a warning is
    issued if many are found."""
    dirs = glob(os.path.join(basechanneldir(channel, cachedir=cachedir),
                             '*' + _DIRNAME_SUFFIX))
    if len(dirs) > 1:
        warnings.warn(("Found multiple raw directories for {}, suggesting "
                       "some sort of cache corruption! matching "
                       "dirs: {}").format(channel, dirs))
    return dirs


def _cacherate(channel, rate, cachedir):
    """Cache the channel sample rate by creating a directory for the raw data
    cache. The name of the raw data cache directory will be the sample rate.
    Return the input rate so that it can be returned by `sample_rate`."""
    if not _matching_raw_dirs(channel, cachedir):
        makedirs(os.path.join(basechanneldir(channel),
                              _dirname_from_rate(rate)))
    return rate


def sample_rate(channel, allow_remote=False, cachedir=DEFAULT_CACHEDIR):
    """Try to find the sample rate (in Hz) of a channel by checking available
    local (and, optionally, potentially slow remote) sources.  Returns an
    astropy quantity representing the sample rate.

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    allow_remote : `bool`, optional, default: `False`
        whether gwpy should try to get the sample_rate for this channel from a
        remote (possibly slow) data source.

    cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
        the directory in which to store cached timeseries data. This directory
        might be checked for existing data on this channel in order to infer
        the sample rate.

    Returns
    -------
    rate : `astropy.units.Quantity`
        the sample rate of this channel.

    Raises
    ------
    IOError
        if the channel's sample rate cannot be found.
    """
    # first see if this is a trend channel
    trend = splitchan(channel)[2]
    if trend is not None:
        if trend == "m-trend":
            return Quantity(1/60., "1/s")
        if trend == "s-trend":
            return Quantity(1., "1/s")
        raise ValueError("Unrecognized trend type: {}".format(trend))
    # if this is a channel object, it will have a sample_rate defined. if it
    # hasn't been explicitly set, it will default to `None`.
    if hasattr(channel, 'sample_rate') and channel.sample_rate is not None:
        return _cacherate(channel, Quantity(channel.sample_rate, "1/s"),
                          cachedir)
    # full data for the base channel name will be stored in a directory whose
    # name is the sample rate followed by a suffix; if we have some data
    # cached, this directory name can be parsed to get the sample rate.
    dirs = [os.path.basename(d) for d in _matching_raw_dirs(channel, cachedir)
            if os.path.isdir(d)]
    if len(dirs) == 1:
        return Quantity(float(dirs[0][:len(_DIRNAME_SUFFIX)]), "1/s")
    if allow_remote:
        try:
            return _cacherate(channel,
                              Quantity(nds_query_info(channel)[1], "1/s"),
                              cachedir)
        except IOError:
            warnings.warn(("Could not find {} sample rate via "
                           "NDS2.").format(channel))
    raise IOError("Can't find sample rate for channel: {}".format(channel))


def indices_to_slices(inds):
    """Takes a list of indices and turns it into a list of start/stop tuples
    (useful for splitting an array into contiguous subintervals).

    Parameters
    ----------
    inds : array-like
        a sorted list of unique positive integers which can be used as indices.
        We expect to have some contiguous subsequences (e.g. `[1,2,3]`) in this
        list that could be replaced by slice notation (e.g. "1:4"
        for that example).

    Returns
    -------
    slices : `list`
        if there are N contiguous subintervals in the list if input indices,
        returns a numpy.ndarray with length 2*N of the format:

        `[(start1, end1), ..., (startN, endN)]`

        Here, the Ith tuple, `(startI, endI)`, could replace the Ith continuous
        subsequence (in the list of indices provided as an argument) via an
        array slice of the form "startI:endI", so that e.g. subsequence
        `[1,2,3]` could be replaced with the tuple `(1,4)` (corresponding to
        slice notation "1:4").

    Examples
    --------
    >>> indices_to_slices([0,1,2,3,9,10,11])
    [(0, 4), (9, 12)]
    """
    if len(inds) == 0:  # pylint: disable=len-as-condition
        return list()
    # cast input as a sorted ndarray and make sure indices are unique ints
    inds, counts = np.unique(inds, return_counts=True)
    if any(inds < 0):
        raise ValueError("Input indices must be positive.")
    if not issubclass(inds.dtype.type, numbers.Integral):
        raise ValueError("Input indices must be integers.")
    if max(counts) != 1:
        raise ValueError("Input indices must be unique.")
    # look for indices in contiguous chunks. find the the indices
    # of the edges of those chunks within the list of non-filler indices.
    change_inds = np.argwhere(inds[1:] != inds[:-1] + 1).flatten()
    # use list comprehension to flatten the list of ends/starts; these are
    # still indices into the list of nonfiller indices rather than indices into
    # the full timeseries itself.
    inner_inds = [
        ind
        for subint in [[i, i+1] for i in change_inds]
        for ind in subint
    ]
    all_inds = np.concatenate([[0], inner_inds, [-1]]).astype(int)
    intervals = inds[all_inds]
    # intervals now looks like [start1, end1-1, ..., startN, endN-1]; reformat
    # it into slice tuples
    return [(intervals[2*k], intervals[2*k+1] + 1)
            for k in range(len(intervals)//2)]


def timeintervals(timeseries, indices, flatten=False, unit=None):
    """Get a list of GPS time intervals corresponding to the provided indices.

    Parameters
    ----------
    timeseries : `gwpy.timeseries.TimeSeriesBase`
        a `TimeSeriesBase` instance with the `times` attribute defined.

    indices : array-like
        a list of indices into `timeseries`. We want to find the GPS times that
        these indices correspond to.

    flatten : `bool`, optional, default: `False`
        whether to flatten the results into a single `Quantity` array.

    unit : `~astropy.units.UnitBase` instance, str, optional
        if provided, return the time intervals as numbers (rather than astropy
        `Quantity` instances) after converting them to the given `unit`.

    Returns
    -------
    intervals : list
        a list of tuples of the form `(start, stop)` corresponding to the
        (half-open) GPS time intervals specified by indices. Useful for finding
        `(start, stop)` times for the sake of data retrieval. If `flatten` is
        `True`, then flatten `intervals` into a single array before returning.
    """
    tim = timeseries.times
    dlt = timeseries.dt
    if unit is not None:
        tim = tim.to(unit).value
        dlt = dlt.to(unit).value
    # have to be careful with the final ending time, which might lie one tick
    # outside of the times contained in `timeseries` (due to the half-open
    # interval notation used by pythonic array slicing).
    res = [(tim[s[0]], tim[s[1]-1] + dlt) for s in indices_to_slices(indices)]
    if flatten:
        if unit is not None:
            return np.array(res).flatten()
        return Quantity([t for interval in res for t in interval])
    return res


def timeintervals_to_indices(timeseries, timeintervals):
    """Inverts `timeintervals`. Get indices in `timeseries` lying within any of
    the half-open intervals specified by `timeintervals`.

    See Also
    --------
    timeintervals
        Inverse function with similar signature.
    """
    out = np.full(timeseries.times.shape, False)
    for start, end in timeintervals:
        out = np.logical_or(out, np.logical_and(timeseries.times >= start,
                                                timeseries.times < end))
    return np.nonzero(out)[0]


def squash_queries(queries, min_duration=MIN_QUERY_SECONDS):
    """Take a list of multi-channel queries intended for a `TimeSeriesBaseDict`
    query method (like `fetch` or `find`) and combine small (i.e. shorter than
    `min_duration`), temporally-adjacent queries so that we don't run a huge
    number of very short contiguous queries (since many short queries are
    slower than fewer long ones, even if the long ones contain some redundant
    information). Useful if the cache is fragmented with very different time
    intervals cached for different channels.

    Parameters
    ----------
    queries : array-like
        a list of queries of the form (channels, start, end) matching the
        function signature of `TimeSeriesBaseDict` query functions like `fetch`
        and `find`, i.e. `channels` is a list of channels and `start` and `end`
        are start an end times of the queries. **The input queries must be
        non-overlapping and sequentially-ordered.**

    min_duration : int, optional, default: `MIN_QUERY_SECONDS`
        roughly speaking, the minimum temporal duration of a query. If two
        queries lie within the same `min_duration` time window, they will be
        combined.

    Returns
    -------
    sqaushed : array-like
        a superset of the data requested in the input list of `queries`, but
        with very short queries combined to reduce the overall number of
        queries (to improve performance). Some the squashed queries will
        contain channels that are already cached for part of their duration,
        but this redundant data fetching is still faster than splitting the
        request across multiple queries.

    Examples
    --------
    >>> squash_queries(
    ...     [DictQuery(['a', 'b'], 0.3, 0.4), DictQuery(['a'], 0.4, 0.5)],
    ...     min_duration=1
    ... )
    [DictQuery(['a', 'b'], 0.3, 0.5)]

    See Also
    --------
    DictQuery
        a nice way of packaging queries that allows them to be conveniently
        combined. Docstring links to more information on GWpy data-retrieval
        methods.
    """
    length = len(queries)
    if any([queries[i+1][1] - queries[i][2] < 0 for i in range(length - 1)]):
        raise ValueError('queries must be non-overlapping and '
                         'sequentially-ordered.')
    if any([not isinstance(q, DictQuery) for q in queries]):
        raise ValueError('queries must be instances of `DictQuery`')
    short_query_inds = [i for i, query in enumerate(queries)
                        if (query[2] - query[1]) < min_duration]
    adjacent_to_next = [i for i in range(length - 1)
                        if queries[i][2] == queries[i+1][1]]
    indices_to_combine = [i for i in adjacent_to_next
                          if i in short_query_inds or i+1 in short_query_inds]
    squashed = list()
    current = queries[0]
    for ind in range(length - 1):
        if ind in indices_to_combine:
            current += queries[ind+1]
        else:
            squashed.append(current)  # can't combine with the next query...
            current = queries[ind+1]  # ...so move on
    squashed.append(current)
    return squashed


def basechanneldir(channel, cachedir=DEFAULT_CACHEDIR):
    """Get the path to the directory containing all the cached data for this
    base channel, stripping away trend extensions.

    Parameters
    ----------
    channel : `~gwpy.detector.Channel`, `str`
        the name of the cached channel.

    cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
        the directory in which to store cached timeseries data.

    Returns
    -------
    path : `str`
        the path to the directory for this base channel.

    See Also
    --------
    splitchan
        a function for parsing channels into a basename plus trend type.
    """
    return os.path.join(cachedir, quote(splitchan(channel)[0]))


def channeldir(channel, cachedir=DEFAULT_CACHEDIR):
    """Get the path to the directory containing all the cached data for this
    specific channel (including its trend extensions).

    Parameters
    ----------
    channel : `~gwpy.detector.Channel`, `str`
        the name of the cached channel.

    cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
        the directory in which to store cached timeseries data.

    Returns
    -------
    path : `str`
        the path to the directory for this channel.

    See Also
    --------
    splitchan
        a function for parsing channels into a basename plus trend type.

    basechanneldir
        gets the directory containing all cached data related to this channel
        (regardless of trend-extension); this is the parent directory of
        `path`.
    """
    trend = splitchan(channel)[2]
    if trend:
        return os.path.join(basechanneldir(channel, cachedir=cachedir), trend)
    return os.path.join(basechanneldir(channel, cachedir=cachedir),
                        _dirname_from_rate(sample_rate(channel)))


def splitchan(channel):
    """Take full channel name and parse the basename and trend type. Inverse
    function of `joinchan`.

    Parameters
    ----------
    channel : `~gwpy.detector.Channel`, `str`
        the name of the cached channel.

    Returns
    -------
    basename : `str`
        the channel name with trend extensions stripped away. If this channel
        is not a trend channel, this will just be the same as `channel`.

    stat : `str`
        the type of statistic being measured, e.g. `max`, `min`, etc. If this
        is not a trend channel, `stat` will be `None`.

    trend : `str`
        `"m-trend"` for minute trends, `"s-trend"` for second trends, `None`
        for non-trend channels.

    Raises
    ------
    ValueError
        if the type of statustic and/or trend type parsed from the given
        channel have unrecognized values.

    Examples
    --------
    >>> splitchan('H1:SYS-TIMING_C_MA_A_PORT_9_UP.max')
    ('H1:SYS-TIMING_C_MA_A_PORT_9_UP', 'max', 'm-trend')

    >>> splitchan('H1:SYS-TIMING_C_MA_A_PORT_9_UP.min,s-trend')
    (''H1:SYS-TIMING_C_MA_A_PORT_9_UP, 'min', 's-trend')

    >>> splitchan('H1:SYS-TIMING_C_MA_A_PORT_9_UP')
    ('H1:SYS-TIMING_C_MA_A_PORT_9_UP', None, None)

    >>> ch = 'H1:SYS-TIMING_C_MA_A_PORT_2_SLAVE_CFC_TIMEDIFF_2.min,s-trend'
    >>> joinchan(*splitchan(ch)) == ch
    True

    See Also
    --------
    joinchan
        the inverse function of `splitchan`.

    TREND_TYPES
        recognized trend types, e.g. 'm-trend', 's-trend'

    TREND_STATISTICS
        recognized trend statistics, e.g. 'min', 'max', 'mean'
    """
    channel = str(channel)
    if '.' not in channel:  # no '.' means no trend extension, i.e. raw channel
        return channel, None, None
    chan, ext = str(channel).split('.')
    if ',' not in ext:  # no ',' in extension implicitly means "m-trend"
        return chan, ext, "m-trend"
    stat, trend = ext.split(',')
    if trend not in TREND_TYPES:
        raise ValueError("unrecognized trend type: {}".format(trend))
    if stat not in TREND_STATISTICS:
        raise ValueError("unrecognized trend statistic: {}".format(stat))
    return chan, stat, trend


def joinchan(basename, stat=None, trend="m-trend"):
    """Get the full channel name as used by GWpy's query methods from the
    basename as well as the statistic being measured as well as the trend type
    (`"m-trend"`, `"s-trend"`, or `None`).

    See Also
    --------
    splitchan
        the inverse function of `joinchan`. See `splitchan` docstring for
        details on the input/output of this function (parameters and return
        values are, of course, reversed).
    """
    if stat is None or trend is None:
        return basename
    if trend == "m-trend":  # m-trend is implicit in GWpy channel names
        return "{}.{}".format(basename, stat)
    if trend not in TREND_TYPES:
        raise ValueError("unrecognized trend type: {}".format(trend))
    return "{}.{},{}".format(basename, stat, trend)


def cacheduration(channel_or_sample_rate):
    """Get the duration (in seconds) that a cache file should last for based on
    a channel or a sample rate for a channel. This duration is chosen so as to
    prevent cache files from being larger than ~4MB when uncompressed. (Since
    this is the minimum amount of data that can be quickly read into memory
    without partial IO, it should be a small chunk of data that even a modest
    computer can quickly load.)

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`, `astropy.units.Quantity`
        the name of the channel to read or a `Channel` object (in which case
        the sample rate will be looked up) or a quantity representing
        the sample rate of some raw channel.

    Returns
    -------
    duration : `astropy.units.Quantity`
        the number of seconds to use per cache file.

    Raises
    ------
    ValueError
        if some assumption about sample rates is violated.

    IOError
        if sample rate information cannot be obtained from the provided channel
        information.

    Examples
    --------
    >>> cacheduration('H1:SYS-TIMING_C_MA_A_PORT_9_UP.max')
    <Quantity 4194304. s>

    >>> cacheduration('H1:SYS-TIMING_C_MA_A_PORT_9_UP.min,s-trend')
    <Quantity 65536. s>

    >>> cacheduration(Quantity('16 Hz'))
    <Quantity 32768. s>
    """
    if isinstance(channel_or_sample_rate, Quantity):
        rate = channel_or_sample_rate.to("1/s").value
    else:
        rate = sample_rate(channel_or_sample_rate).to("1/s").value
        # trend statistics are stored together and hence need extra space
        if splitchan(channel_or_sample_rate)[2] is not None:
            rate *= len(TREND_STATISTICS)
    # include a fudge factor to account for numerical precision of rate
    if rate > MAX_SAMPLE_RATE + 1.:
        raise ValueError(("Sample rate too high, violates GWpy assumptions "
                          "about GW data! Sample rate: {}").format(rate))
    # we want 4MB in memory on load = 2**22 bytes = 2**19 doubles
    return Quantity(2**(19 - int(np.ceil(np.log2(rate)))), "s")


CacheInfo = namedtuple("CacheInfo", ("path", "start", "end"))


def cachefiles(channel, start, end, cachedir=DEFAULT_CACHEDIR):
    """Get a temporally-ordered list of cached filenames along with the times
    they correspond to for a given channel.

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data, any input parseable by
        `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data, any input parseable by
        `~gwpy.time.to_gps` is fine

    cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
        the directory in which to store cached timeseries data.

    Returns
    -------
    cacheinfos : `tuple`, `type(None)`
        a tuple of `CacheInfo` tuples. The files are in temporal order such
        that they can be batch-loaded and blindly appended (assuming they all
        exist). If sample rate information cannot be obtained from the provided
        channel information, returns `None`.

    Raises
    ------
    ValueError
        if some assumption about sample rates is violated.
    """
    try:
        duration = int(cacheduration(channel).to("s").value)
    except IOError:
        return None
    start_ind = int(np.floor((to_gps(start).ns() / 1e9) / duration))
    end_ind = int(np.ceil((to_gps(end).ns() / 1e9) / duration))
    edges = np.arange(start_ind, end_ind+1) * duration
    chandir = channeldir(channel, cachedir=cachedir)
    return tuple([CacheInfo(_cachepath(chandir, edges[i], edges[i+1]),
                            edges[i], edges[i+1])
                  for i in range(len(edges)-1)])


def _cachepath(chandir, start, end):
    """Get a full path to a cachefile located in channel directory `chandir`
    (as returned by `channeldir`) covering times ranging from GPS times `start`
    to `end` (both of which must be integers)."""
    subdir = int(np.floor(start / SUBDIR_DURATION) * SUBDIR_DURATION)
    return os.path.join(chandir, str(subdir), "{}-{}.hdf5".format(start, end))


def placeholder(seed, start, end, missing=DEFAULT_MISSING_VALUE):
    """Create a placeholder `TimeSeriesBase` or `TimeSeriesBaseDict` (or
    subclasses) instance between the times of `start` and `end` and fill it
    with a placeholder value defined by `missing`.

    Parameters
    ----------
    seed : `TimeSeriesBase`, `TimeSeriesBaseDict`
        an input timeseries data container. We will make our placeholder match
        the channel info contained in `seed` so that their data can readily be
        combined.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data, any input parseable by
        `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data, any input parseable by
        `~gwpy.time.to_gps` is fine

    missing : optional, default: `DEFAULT_MISSING_VALUE`
        **NB: You should not manually set this unless you really know what you
        are doing, since it affects how missing data in the cache is
        interpreted.** This is the pad value used in cached timeseries data to
        represent data that has not yet been cached yet.

    Returns
    -------
    placeholder : `TimeSeriesBase`, `TimeSeriesBaseDict`
        an instance with the same class as `seed` covering the same channels
        between times `start` and `end` but filled with padded values defined
        by `missing`.
    """
    # if this is some sort of TimeSeriesDict, make a placeholder for each
    # component timeseries recursively and then return the full dict.
    start = Quantity(to_gps(start).ns(), "ns")
    end = Quantity(to_gps(end).ns(), "ns")
    if isinstance(seed, dict):
        result = dict()
        for chan, timeseries in seed.items():
            result[chan] = placeholder(timeseries, start, end, missing=missing)
        return type(seed)(result)
    length = int(round(((end - start) / seed.dt).to("").value))
    return type(seed)(
        np.full((length,), missing),
        unit=seed.unit,
        t0=start,
        dt=seed.dt,
        name=seed.name,
        channel=seed.channel,
        dtype=seed.dtype,
        copy=False
    )


def read(cls, channels, start, end, cachedir=DEFAULT_CACHEDIR,
         missing=DEFAULT_MISSING_VALUE):
    """Read in as much timeseries data as possible from cache. If cached data
    is not available, still try to return a partially-complete timeseries,
    where any missing values are filled in with `missing` as the pad value.

    Parameters
    ----------
    cls : `type`
        the `TimeSeriesBaseDict` subclass to try to read values into.

    channels : array-like
        the name of the channel to read as a string, or a
        `gwpy.detector.Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data, any input parseable by
        `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data, any input parseable by
        `~gwpy.time.to_gps` is fine

    cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
        the directory in which to store cached timeseries data.

    missing : optional, default: `DEFAULT_MISSING_VALUE`
        **NB: You should not manually set this unless you really know what you
        are doing, since it affects how missing data in the cache is
        interpreted.** This is the pad value used in cached timeseries data to
        represent data that has not yet been cached yet.

    Returns
    -------
    loaded : `dict`
        a `dict` (with channel names as keys and `CacheableTimeSeries`
        instances as values) containing as many requested channels as could be
        *partially recovered* from the cache, filling the requested time
        interval, with missing values padded using the value provided in the
        `missing` parameter. Note that missing values indicated that the value
        is missing *from the cache* and should therefore be filled in using
        some other method. This `dict` instance can be passed to
        `CacheableTimeSeriesDict` (or similar) to create an equivalent instance
        of that class.

    failed : `list`
        a list of channel names that could not even be initialized as
        partially-filled timeseries (e.g. because a sample rate could not be
        calculated).

    See Also
    --------
    TimeSeries.missing
        for getting a masking array of boolean values with `True` values
        indicating missing indices.
    """
    start = Quantity(to_gps(start).ns(), "ns").to("s")
    end = Quantity(to_gps(end).ns(), "ns").to("s")
    cacheinfodict = flip((c, cachefiles(c, start, end, cachedir=cachedir))
                         for c in channels)
    seeds = {info: [seed for seed in info if os.path.isfile(seed.path)]
             for info in cacheinfodict}
    failed = sum((cacheinfodict.pop(i) for i in seeds if not seeds[i]),
                 cacheinfodict.get(None, list()))
    loaded = dict()
    for cacheinfos in cacheinfodict:
        seedinfo = seeds[cacheinfos][0]
        seed = cls.read(seedinfo.path)
        out = _read_or_placeholder(cls, cacheinfos[0], seedinfo, seed, missing)
        for dat in cacheinfos[1:]:
            out.append(_read_or_placeholder(cls, dat, seedinfo, seed, missing))
        for chan, tseries in out.items():
            loaded[str(chan)] = tseries[
                slice(*tseries.times.searchsorted((start, end), side='left'))
            ].copy()
    return loaded, failed


def _read_or_placeholder(cls, cacheinfo, seedinfo, seed, missing):
    """Try reading `cachefile` as class `cls`. If `cachefile` doesn't exist,
    make a placeholder from `seed`, `start`, `end`, and `missing`. Don't reload
    `seed` if it has already been loaded and is being encountered again."""
    if cacheinfo == seedinfo:
        return seed
    if os.path.isfile(cacheinfo.path):
        return cls.read(cacheinfo.path)
    return _trend_placeholder(seed, cacheinfo.start, cacheinfo.end,
                              missing=missing)


def write(tseriesdict, cachedir=DEFAULT_CACHEDIR,
          missing=DEFAULT_MISSING_VALUE):
    """Write the data contained in `tseriesdict` to the correct cache files
    using `tseriesdict.write`.

    Parameters
    ----------
    tseriesdict : `TimeSeriesBaseDict`
        a `TimeSeriesBaseDict` instance with a working `write` method.

    cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
        the directory in which to store cached timeseries data.

    missing : optional, default: `DEFAULT_MISSING_VALUE`
        **NB: You should not manually set this unless you really know what you
        are doing, since it affects how missing data in the cache is
        interpreted.** This is the pad value used in cached timeseries data to
        represent data that has not yet been cached yet.
    """
    cls = type(tseriesdict)
    cacheinfodict = {c: cachefiles(c, tseriesdict[c].t0,
                                   tseriesdict[c].times[-1], cachedir=cachedir)
                     for c in tseriesdict}
    allfiles = {f for chan in cacheinfodict for f in cacheinfodict[chan]}
    file_to_chan = {cachefile: cls() for cachefile in allfiles}
    for chan, files in cacheinfodict.items():
        for cachefile in files:
            start = Quantity(cachefile.start, "s")
            end = Quantity(cachefile.end, "s")
            file_to_chan[cachefile][chan] = tseriesdict[chan]
    loaded = {c: _read_or_placeholder(cls, c, None, file_to_chan[cf], missing)
              for c in file_to_chan}
    return file_to_chan, loaded
    for cf in loaded:
        loaded[cf].update(file_to_chan[cf])
    # TODO chunk timeseries by file into smaller dicts
    # TODO for tseriesdicts not filling the file, read the file first and
    # update values with values in tseries (by calling read)
    # TODO write these now-filled tseries dicts to file
    # TODO implement


def _trend_placeholder(seed, start, end, missing=DEFAULT_MISSING_VALUE):
    """Same interface as `placeholder`, but make sure that `seed` contains a
    full set of placeholders for every possible trend-type in `seed`, which can
    be a `TimeSeriesBase` or `TimeSeriesBaseDict` instance."""
    if not isinstance(seed, dict):
        raise ValueError("`seed` must be instance of dict. Got: "
                         "{}".format(seed))
    splits = [splitchan(t.channel) for t in seed.values()]
    basenames = set(s[0] for s in splits)
    stats = [s[1] for s in splits]
    trend_types = set(s[2] for s in splits)
    if len(basenames) != 1:
        raise ValueError("Seed must contain only one basechannel. Got: "
                         "{}".format(basenames))
    if len(trend_types) != 1:
        raise ValueError("Seed must contain only one trend type. Got: "
                         "{}".format(trend_types))
    basename = basenames.pop()
    trend = trend_types.pop()
    if trend is None:
        return placeholder(seed, start, end, missing=missing)
    chans = [joinchan(basename, s, trend) for s in TREND_STATISTICS]
    firsttseries = list(seed.values())[0]
    # since placeholder doesn't care about actual data content, just feed it
    # the same timeseries for all keys
    newseed = type(seed)((c, firsttseries) for c in chans)
    return placeholder(newseed, start, end, missing=missing)


def cacheable(func):
    """A decorator for a `TimeSeries` or `TimeSeriesDict`-fetching class-method
    `func` that adds the ability to read from or write to a user-specific
    cache.

    Parameters
    ----------

    func
        a classmethod for a `TimeSeriesDict`-related class that fetches data
        and accepts `(cls, channels, start, end, pad, **kwargs)` in its
        function interface.

    Returns
    -------

    wrapper
        a wrapper around `func` that adds the following keyword arguments to
        the interface of `func`:

        cache : `bool`, optional, default: `None`
            whether to try using the cache to get data, and whether to write
            newly-downloaded data to the cache. Default caching behavior is
            dynamically determined by checking environmental variable
            `GWPY_CACHE`; if set to `GWPY_CACHE=1`, caching is on by default.
            Otherwise, it is off by default.

        cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
            the directory in which to store cached timeseries data.
    """
    def wrapper(cls, channels, start, end, pad=None, cache=None,
                cachedir=DEFAULT_CACHEDIR, **kwargs):
        """A wrapper adding caching functionality to a GWpy data fetching
        classmethod."""
        if cache is None:
            cache = _get_default_caching_behavior()
        # get the TimeSeriesBaseDict class if not already a subclass thereof
        get_dictclass = hasattr(cls, "DictClass")
        if get_dictclass:
            cls = cls.DictClass
            channels = [channels]
        if cache:
            loaded, failed = read(cls, channels, start, end, cachedir=cachedir)
            result = CacheableTimeSeriesDict(loaded)
            for query in squash_queries(result.missing_queries()):
                newdata = func(cls, query.channels, query.start, query.end,
                               pad=pad, **kwargs)
                result.update(newdata)
                write(newdata, cachedir=cachedir)
            if failed:
                result.update(func(cls, failed, start, end, pad=pad, **kwargs))
            if get_dictclass:
                return result[channels[0]]
            return result
        return func(cls, channels, start, end, pad=pad, **kwargs)
    return wrapper


class CacheableTimeSeries(TimeSeriesBase):
    # TODO

    def replace_in_place(self, other, **kwargs):
        """Same as `replace`, but if the times found in `other` completely
        overlap with `self`, then the values will be updated **in place** to
        save memory and time and the updated `CacheableTimeSeries` instance
        will be returned. **Only use this method if you understand the
        lifecycle of this instance and know that it can be safely mutated.**

        See Also
        --------
        CacheableTimeSeries.replace
            does the exact same thing as this method but without ever mutating
            the input instances.
        """
        if self.t0 <= other.t0 and self.times[-1] >= other.times[-1]:
            start = np.argwhere(self.times == other.t0)[0][0]
            end = np.argwhere(self.times == other.times[-1])[0][0]
            self[start:end+1] = other
            return self
        return self.replace(other, **kwargs)

    # pylint: disable=len-as-condition
    def replace(self, other, **kwargs):
        """Return a copy of this `TimeSeriesBaseDict` instance updated with
        values contained in `other`. By default, assumes that both input dicts
        correspond to overlapping time intervals (since discontinuous
        `TimeSeriesBase` instances cannot be appended by default). For any
        times appearing in *both* dicts, the value from `other` will be used.
        This method can be used to iteratively fill in the values of a
        `TimeSeries` instance.

        Parameters
        ----------
        self : `CacheableTimeSeries`
            a `CacheableTimeSeries` whose values we want to update and/or
            append to.

        other : `CacheableTimeSeries`
            the other `CacheableTimeSeries` whose values we want to incorporate
            into `self`. Must be contiguous or overlapping with `self` (or else
            we'd have to worry about how to pad missing values).

        kwargs
            keyword arguments to pass to `self.append`.

        See Also
        --------
        gwpy.timeseries.TimeSeriesBase.append
            the method used to append two timeseries.
        """
        before = np.argwhere(self.times < other.times[0]).flatten()
        after = np.argwhere(self.times > other.times[-1]).flatten()
        if len(before) == 0 and len(after) == 0:  # other totally covers self
            return other.copy()
        if len(before) == 0:  # part of self is after other
            return other.copy().append(self[after], **kwargs)
        if len(after) == 0:  # part of self is before other
            return self[before].copy().append(other, **kwargs)
        return self[before].copy().append(other, **kwargs).append(self[after],
                                                                  **kwargs)


class CacheableTimeSeriesDict(TimeSeriesBaseDict):
    # TODO

    def update(self, other):
        """Update the contents of this `TimeSeriesBaseDict` instance **in
        place** with the contents of the `other` instance. Assumes that `other`
        is a subclass of `TimeSeriesBaseDict` and tries to add new channels to
        this instance while also updating existing channels with any new data
        found in the corresponding channels in other. For instance, if channel
        'foo' exists in both instances, then update 'foo' in this instance
        using `CacheableTimeSeriesDict.update(self, other)`.

        Like the builtin `dict.update` method, this method has no return value.

        Parameters
        ----------
        self : `gwpy.timeseries.CacheableTimeSeriesDict`
            a `CacheableTimeSeriesDict` whose channels we want to update or
            add data to.

        other : `gwpy.timeseries.TimeSeriesBaseDict`
            a `TimeSeriesBaseDict` containing data that we want to insert into
            `self`. For channels already contained in `self`, any overlapping
            times will be replaced with data contained in `other`. For channels
            *not* contained in `self`, the timeseries in `other` will simply be
            added to `self`.
        """
        for key in other.keys():
            if key in self:
                self[key] = self[key].replace(other[key])
            else:
                self[key] = other[key]

    def missing_queries(self, test=np.isnan, param=()):
        """Take an input `TimeSeriesBaseDict` instance and return a list of
        `TimeSeriesBaseDict` query parameters that can be used to efficiently
        retrieve the missing values via `TimeSeriesBaseDict.get` and related
        methods.

        Parameters
        ----------
        self : `gwpy.timeseries.TimeSeriesBaseDict`
            a `TimeSeriesBaseDict` instance with potentially missing values
            (e.g.  values that have not yet been stored in the local cache) as
            defined by `test`. *(Note that the channels in `self` need
            not have the same missing time intervals; this function will
            automatically generate queries for missing data that avoid
            redundant data fetching and group channel queries together to
            maximize speed.)*

        test : optional, default: `np.isnan`
            a function that will return a boolean-type array with the same
            shape as `timeseries` where `True` values indicate that the
            corresponding values in `timeseries` are, in fact, missing.

        param : optional
            parameters to pass to the `test` function, e.g. a constant value
            that is used as padding for missing values in an equality check.
            These will be passed as positional args to `test`; the first
            argument passed to `test` will be a timeseries.

        Returns
        -------
        queries : list
            a list of `DictQuery` instnaces describing the queries to be
            performed.

        See Also
        --------
        DictQuery
            a nice way of packaging queries that allows them to be conveniently
            combined. Docstring links to more information on GWpy
            data-retrieval methods.
        """
        # get indices and time intervals of missing values for each channel.
        # we will split the missing time intervals along the boundaries of each
        # interval and then regroup everything later
        ind = {c: np.nonzero(test(t, *param))[0] for c, t in self.items()}
        time = {c: timeintervals(t, ind[c], unit="s") for c, t in self.items()}
        edges = np.unique([v for c in time for i in time[c] for v in i])
        missing_channels_at_time = dict()
        # now see which time intervals correspond to which channel combinations
        for chan in self:
            for start, end in time[chan]:
                cut_inds = np.logical_and(edges > start, edges < end)
                bounds = [start] + list(edges[cut_inds]) + [end]
                splt = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
                for interval in splt:
                    if interval not in missing_channels_at_time:
                        missing_channels_at_time[interval] = [chan]
                    else:
                        missing_channels_at_time[interval].append(chan)
        return [DictQuery(missing_channels_at_time[i], i[0], i[1])
                for i in missing_channels_at_time]

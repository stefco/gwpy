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

**None of these objects are really designed to be used other than as bases for
user-facing objects.**
"""

from __future__ import print_function

import os
import numbers
from collections import namedtuple

import numpy as np
from numpy import nan
from astropy.units import Quantity

from ..core import TimeSeriesBase, TimeSeriesBaseDict

DEFAULT_CACHEDIR = os.path.expanduser("~/.local/share/gwpy/cache")
DEFAULT_MISSING_VALUE = nan
# TODO Optimize the minimum query length with some testing; 1s is just a sane
# lower-bound, but it will probably be closer to 64s (smallest common frame
# duration).
MIN_QUERY_SECONDS = 1
MIN_DT = 2**-16  # shortest used dt between samples; used for rounding

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


# TODO increase the number of possible ways to check for sample rate, including
# (possibly) seeing if we are currently on a dataframe-holding server that has
# datafind available.
def sample_rate(channel, allow_remote=False):
    """Try to find the sample rate (in Hz) of a channel, first by checking for
    a cached sample rate value, second by seeing if a sample rate can be
    retrieved from other (possibly slower or remote) sources. Returns a float
    representing the sample rate.

    Parameters
    ----------

    channel : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    allow_remote : `bool`, optional, default: `False`
        whether gwpy should try to get the sample_rate for this channel from a
        remote (possibly slow) data source.
    """
    # TODO implement


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
    if len(inds) == 0:
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


def timeintervals(timeseries, indices):
    """Get a list of GPS time intervals corresponding to the provided indices.

    Parameters
    ----------
    timeseries : `gwpy.timeseries.TimeSeriesBase`
        a `TimeSeriesBase` instance with the `times` attribute defined.

    indices : array-like
        a list of indices into `timeseries`. We want to find the GPS times that
        these indices correspond to.

    Returns
    -------
    intervals : list
        a list of tuples of the form `(start, stop)` corresponding to the
        (half-open) GPS time intervals specified by indices. Useful for finding
        `(start, stop)` times for the sake of data retrieval.
    """
    times = timeseries.times
    dt = timeseries.dt  # pylint: disable=invalid-name
    slices = indices_to_slices(indices)
    # have to be careful with the final ending time, which might lie one tick
    # outside of the times contained in `timeseries` (due to the half-open
    # interval notation used by pythonic array slicing).
    return [(times[s[0]], times[s[1]-1] + dt) for s in slices]


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


def read(channels, start, end, cachedir=DEFAULT_CACHEDIR,
         missing=DEFAULT_MISSING_VALUE):
    """Read in as much timeseries data as possible from cache. If cached data
    is not available, still try to return a partially-complete timeseries,
    where any missing values are filled in with `missing` as the pad value.

    Parameters
    ----------
    channels : array-like
        the name of the channel to read as a string, or a
        `gwpy.detector.Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

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

    failed : array-like
        a list of channel names that could not even be initialized as
        partially-filled timeseries (e.g. because a sample rate could not be
        calculated).

    See Also
    --------
    TimeSeries.missing
        for getting a masking array of boolean values with `True` values
        indicating missing indices.
    """
    # TODO implement


def cacheabledict(func):
    """A decorator for `TimeSeriesDict`-fetching class-method `func` that adds
    the ability to read from or write to a user-specific cache.

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

        cache : `bool`, optional, default: `True`
            whether to try using the cache to get data, and whether to write
            newly-downloaded data to the cache.

        cachedir : `str`, optional, default: `DEFAULT_CACHEDIR`
            the directory in which to store cached timeseries data.
    """
    def wrapper(cls, channels, start, end, pad=None, cache=True,
                cachedir=DEFAULT_CACHEDIR, **kwargs):
        if cache:
            loaded, failed = read(channels, start, end, cachedir=cachedir)
            result = CacheableTimeSeriesDict(loaded)
            for query in squash_queries(result.missing_queries()):
                newdata = func(cls, query.channels, query.start, query.end,
                               pad=pad, **kwargs)
                result.update(newdata)
                write(newdata, cachedir=cachedir)
            if failed:
                result.update(func(cls, failed, start, end, pad=pad, **kwargs))
            return result
        return func(cls, channels, start, end, pad=pad, **kwargs)
    return wrapper


class CacheableTimeSeries(TimeSeriesBase):
    # TODO

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

    def missing_queries(self, test=np.equal, param=DEFAULT_MISSING_VALUE):
        """Take an input timeseries dict and return a list of
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

        test : optional, default: `np.equal`
            a function with signature `test(timeseries, param)` that will
            return a boolean-type array with the same shape as `timeseries`
            where `True` values indicate that the corresponding values in
            `timeseries` are, in fact, missing.

        param : optional, default: `DEFAULT_MISSING_VALUE`
            a parameter to pass to the `test` function, e.g. a constant value
            that is used as padding for missing values in an equality check.

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
        if not isinstance(self, dict):
            raise ValueError("`self` must be an instance of `dict`.")
        # get indices and time intervals of missing values for each channel
        indices = dict()
        times = dict()
        # we will split the missing time intervals along the boundaries of each
        # interval and then regroup everything later
        boundaries = tuple()
        for chan, tseries in self.items():
            indices[chan] = np.argwhere(test(tseries, param)).flatten()
            times[chan] = timeintervals(tseries, indices[chan])
            boundaries += sum(times[chan], ())
        # collect all boundaries together and eliminate duplicate values
        boundaries = np.unique(Quantity(boundaries))
        missing_channels_at_time = dict()
        # now see which time intervals correspond to which channel combinations
        for chan in self:
            for intrvl in times[chan]:
                cut_inds = np.logical_and(boundaries > intrvl[0],
                                          boundaries < intrvl[1])
                bounds = [intrvl[0]] + list(boundaries[cut_inds]) + [intrvl[1]]
                splits = [(bounds[i], bounds[i+1])
                          for i in range(len(bounds)-1)]
                for split_interval in splits:
                    if split_interval not in missing_channels_at_time:
                        missing_channels_at_time[split_interval] = [chan]
                    else:
                        missing_channels_at_time[split_interval].append(chan)
        return [DictQuery(missing_channels_at_time[i], i[0], i[1])
                for i in missing_channels_at_time]

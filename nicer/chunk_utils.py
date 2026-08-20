import numpy as np


def resolve_gti_extension(hdulist, hdr, requested_extname, events_data=None):
    """Resolve the GTI extension to use, including XMM STDGTI fallback."""
    available = [h.name for h in hdulist]

    if requested_extname in available:
        return requested_extname

    telescop = hdr.get("TELESCOP", "")
    if str(telescop).startswith("XMM"):
        stdgti_matches = [name for name in available if name.upper().startswith("STDGTI")]

        if not stdgti_matches:
            raise RuntimeError(
                f"No '{requested_extname}' extension found, and no STDGTI* "
                f"extension found either for this XMM file. Available extensions: {available}"
            )

        if len(stdgti_matches) == 1:
            return stdgti_matches[0]

        if events_data is not None and "CCDNR" in events_data.names:
            ccdnrs = events_data["CCDNR"]
            values, counts = np.unique(ccdnrs, return_counts=True)
            dominant_ccd = int(values[np.argmax(counts)])
            candidate = f"STDGTI{dominant_ccd:02d}"
            if candidate in stdgti_matches:
                return candidate

        return stdgti_matches[0]

    return requested_extname


def _get_mjdref_keywords(hdr):
    """Return (mjdrefi, mjdreff, timezero) for FITS TIME keywords."""
    if "MJDREFI" in hdr and "MJDREFF" in hdr:
        mjdrefi = hdr["MJDREFI"]
        mjdreff = hdr["MJDREFF"]
    elif "MJDREF" in hdr:
        mjdrefi = int(hdr["MJDREF"])
        mjdreff = hdr["MJDREF"] - mjdrefi
    else:
        raise RuntimeError("No MJDREF/MJDREFI+MJDREFF keywords found in header")
    timezero = hdr.get("TIMEZERO", 0.0)
    return mjdrefi, mjdreff, timezero


def met_to_mjd_approx(met, mjdrefi, mjdreff, timezero):
    """Approximate MET -> MJD conversion for chunk boundary planning."""
    return mjdrefi + mjdreff + (timezero + met) / 86400.0


def tint_gti_groups(gti_t0, gti_t1, tint, maxint, dice=False):
    """Return GTIs (optionally diced) and slices contributing to each TOA.

    The grouping is shared by chunked and non-chunked processing so a chunk
    boundary can never change the exposure or photon selection of a TOA.
    """
    gti_t0 = np.asarray(gti_t0, dtype=float)
    gti_t1 = np.asarray(gti_t1, dtype=float)
    if dice:
        starts, stops = [], []
        for start, stop in zip(gti_t0, gti_t1):
            pieces = max(1, int(np.floor((stop - start) / tint)) + 1)
            edges = np.linspace(start, stop, pieces + 1)
            starts.extend(edges[:-1])
            stops.extend(edges[1:])
        gti_t0, gti_t1 = np.asarray(starts), np.asarray(stops)

    groups, start_index, exposure = [], 0, 0.0
    for index, (start, stop) in enumerate(zip(gti_t0, gti_t1)):
        exposure += stop - start
        if (
            exposure >= tint
            or stop - gti_t0[start_index] > maxint
            or index == len(gti_t0) - 1
        ):
            groups.append(slice(start_index, index + 1))
            start_index, exposure = index + 1, 0.0
    return gti_t0, gti_t1, groups


def plan_chunks_by_gti(mets, gti_t0, gti_t1, chunk_events, groups=None):
    """Group consecutive GTIs into chunks of about ``chunk_events`` photons.

    If ``groups`` is supplied, it contains indivisible GTI slices (one per
    requested TOA). Chunks will only end between groups, preserving results.
    """
    if chunk_events is None:
        return []
    if chunk_events <= 0:
        raise ValueError("chunk_events must be a positive integer")

    units = groups or [slice(index, index + 1) for index in range(len(gti_t0))]
    chunks = []
    i0 = 0
    current_count = 0

    for unit_index, group in enumerate(units):
        c0, c1 = np.searchsorted(mets, [gti_t0[group.start], gti_t1[group.stop - 1]])
        current_count += c1 - c0

        if current_count >= chunk_events or unit_index == len(units) - 1:
            chunks.append(
                {
                    "gti_slice": slice(units[i0].start, group.stop),
                    "group_slices": units[i0 : unit_index + 1],
                    "met_start": gti_t0[units[i0].start],
                    "met_stop": gti_t1[group.stop - 1],
                    "n_events_approx": current_count,
                }
            )
            i0 = unit_index + 1
            current_count = 0

    return chunks


def chunk_boundary_mjd(mets, met_start, met_stop, mjdref_args, pad_events=1):
    """Convert a chunk's MET range into approximate MJD bounds."""
    idx_lo = np.searchsorted(mets, met_start)
    idx_hi = np.searchsorted(mets, met_stop)

    if idx_lo > 0:
        met_lo_bound = 0.5 * (mets[idx_lo - 1] + mets[idx_lo])
    else:
        met_lo_bound = met_start - 1.0

    if idx_hi < len(mets):
        met_hi_bound = 0.5 * (mets[idx_hi - 1] + mets[idx_hi])
    else:
        met_hi_bound = met_stop + 1.0

    minmjd = met_to_mjd_approx(met_lo_bound, *mjdref_args)
    maxmjd = met_to_mjd_approx(met_hi_bound, *mjdref_args)
    return minmjd, maxmjd


def build_event_loader_kwargs(ephem, planets, include_bipm, minmjd, maxmjd):
    """Return the common keyword arguments used by the PINT event loaders."""
    return {
        "ephem": ephem,
        "planets": planets,
        "include_bipm": include_bipm,
        "minmjd": minmjd,
        "maxmjd": maxmjd,
    }

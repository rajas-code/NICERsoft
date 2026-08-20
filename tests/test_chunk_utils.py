import numpy as np
from astropy.io import fits

from nicer.chunk_utils import (
    build_event_loader_kwargs,
    plan_chunks_by_gti,
    resolve_gti_extension,
    tint_gti_groups,
)


def test_plan_chunks_by_gti_groups_consecutive_gtis_without_splitting_them():
    mets = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    gti_t0 = np.array([0.0, 4.0])
    gti_t1 = np.array([3.0, 6.0])

    chunks = plan_chunks_by_gti(mets, gti_t0, gti_t1, chunk_events=3)

    assert len(chunks) == 2
    assert chunks[0]["gti_slice"] == slice(0, 1)
    assert chunks[0]["n_events_approx"] == 3
    assert chunks[1]["gti_slice"] == slice(1, 2)
    assert chunks[1]["n_events_approx"] == 2


def test_build_event_loader_kwargs_preserves_bounds_and_flags():
    kwargs = build_event_loader_kwargs(
        ephem="DE421",
        planets=True,
        include_bipm=True,
        minmjd=50000.0,
        maxmjd=51000.0,
    )

    assert kwargs["ephem"] == "DE421"
    assert kwargs["planets"] is True
    assert kwargs["include_bipm"] is True
    assert kwargs["minmjd"] == 50000.0
    assert kwargs["maxmjd"] == 51000.0


def test_resolve_gti_extension_uses_stdgti_for_xmm_files():
    data = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", array=np.array([1.0])),
            fits.Column(name="STOP", format="D", array=np.array([2.0])),
        ]
    )
    stdgti = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", array=np.array([3.0])),
            fits.Column(name="STOP", format="D", array=np.array([4.0])),
        ],
        name="STDGTI01",
    )
    hdul = fits.HDUList([fits.PrimaryHDU(), data, stdgti])
    hdr = {"TELESCOP": "XMM-Newton"}

    assert resolve_gti_extension(hdul, hdr, "GTI") == "STDGTI01"


def test_chunk_boundaries_do_not_split_tint_toa_groups():
    mets = np.arange(12.0)
    starts = np.array([0.0, 3.0, 6.0, 9.0])
    stops = starts + 3.0
    starts, stops, groups = tint_gti_groups(starts, stops, tint=6.0, maxint=100.0)

    chunks = plan_chunks_by_gti(mets, starts, stops, chunk_events=3, groups=groups)

    assert groups == [slice(0, 2), slice(2, 4)]
    assert [chunk["group_slices"] for chunk in chunks] == [[slice(0, 2)], [slice(2, 4)]]

# Citation verification and publication versions

Checked 2026-11-01 using the literature-review skill's `verify_citations.py`, with primary-source and registry reconciliation. The [verification report](citation_verification.json) records the script hash, per-paper results and reused reviewer artifacts.

All **54 assigned DOI handles resolved**. Crossref or DataCite metadata titles matched all 54 selected titles after case, punctuation and whitespace normalization. The other **10 primary URLs returned successful HEAD responses** and their titles, dates and evidence locations were inspected in the theory extraction. A reachable URL or registered DOI is a bibliographic check, not validation of a theorem or empirical result.

The bridge reviewer checked 17 DOIs; the central pass reused those receipts and the [metadata addendum](agents/bridge_citation_verification_addendum.json). Empty metadata responses were retried against Crossref and then DataCite where necessary. The [follow-up responses](raw/citation_metadata_followup.json) are retained. No primary bibliographic identity remains unresolved; the Istat manuscript is deliberately undated because no publication year was verified.

| Paper or group | Version/date decision |
|---|---|
| Pamona | Online 16 August 2021; January 2022 issue. Preferred citation year 2022. Crossref's 2021 online date is compatible. |
| SIMBA | Online 29 May 2023; June 2024 issue. Preferred citation year corrected from 2023 to 2024. Both dates remain in the extraction. |
| scMRDR | NeurIPS 2025 proceedings DOI is canonical; relevant methods inspected in January 2026 arXiv v2. |
| Sturma; Timilsina; Xi | Canonical NeurIPS proceedings identities checked; preprint versions are not counted as additional papers. |
| Sun biomedical causal representations | Canonical ICLR 2025 paper; full text/theorem statements inspected. |
| Tong temporal OT/GRN | 2026 chapter publication in ECML PKDD 2024 proceedings; detailed methods inspected in the accessible 2024 preprint. |
| Liu multitask benchmark | 2025 Registered Report; its 2024 preprint is a related version, not another selected study. |
| Koo statistical matching | 2026 arXiv preprint, explicitly labeled; DOI is a DataCite arXiv identifier. |
| D'Orazio statistical matching | Verified author manuscript at Istat; publication year and DOI not invented. |

The [paper ledger](PAPER_EVIDENCE.json) preserves evidence depth separately from citation validity. Five records remain abstract/indexed-primary-result limited even though their bibliographic identities are verified. Funding and conflict declarations were not extracted systematically.

def extract_features(trial):
    ps = trial.get("protocolSection", {})
    cm = ps.get("conditionBrowseModule", {})
    
    # Modules for added covariates
    design = ps.get("designModule", {})
    status = ps.get("statusModule", {})
    sponsor = ps.get("sponsorCollaboratorsModule", {})
    oversight = ps.get("oversightModule", {})
    identification = ps.get("identificationModule", {})

    return {
        # Basic Identifiers
        "nct_id": identification.get("nctId"),
        "title": identification.get("briefTitle"),

        # Disease Grouping
        "conditions": ps.get("conditionsModule", {}).get("conditions", []),

        # Trial Design & Complexity
        "study_type": design.get("studyType"),
        "phase": design.get("phases", [None])[0],
        "masking": design.get("designInfo", {}).get("maskingInfo", {}).get("masking"),
        "randomization": design.get("designInfo", {}).get("allocation"),
        "intervention_model": design.get("designInfo", {}).get("interventionModel"),
        "primary_purpose": design.get("designInfo", {}).get("primaryPurpose"),  # New
        "arm_count": len(ps.get("armsInterventionsModule", {}).get("interventions", [])),

        # Interventions
        "interventions": [
            {"type": iv.get("type"), "name": iv.get("name")}
            for iv in ps.get("armsInterventionsModule", {}).get("interventions", [])
        ],

        # Eligibility & Recruitment
        "eligibility_criteria": ps.get("eligibilityModule", {}).get("eligibilityCriteria"),
        "minimum_age": ps.get("eligibilityModule", {}).get("minimumAge"),
        "maximum_age": ps.get("eligibilityModule", {}).get("maximumAge"),
        "sex": ps.get("eligibilityModule", {}).get("sex"),
        "healthy_volunteers": ps.get("eligibilityModule", {}).get("healthyVolunteers"),

        # Outcomes & Scope
        "primary_outcomes": [
            o.get("measure") for o in ps.get("outcomesModule", {}).get("primaryOutcomes", [])
        ],
        "secondary_outcomes": [
            o.get("measure") for o in ps.get("outcomesModule", {}).get("secondaryOutcomes", [])
        ],
        "status": status.get("overallStatus"),
        "start_date": status.get("startDateStruct", {}).get("date"),
        "completion_date": status.get("completionDateStruct", {}).get("date"),

        # Sponsorship & Sites
        "sponsor_name": sponsor.get("leadSponsor", {}).get("name"),
        "sponsor_class": sponsor.get("leadSponsor", {}).get("class"),
        "collaborators": [c.get("name") for c in sponsor.get("collaborators", [])],
        "collaborator_count": len(sponsor.get("collaborators", [])),
        "site_count": len(ps.get("contactsLocationsModule", {}).get("locations", [])),

        # Narrative Summary
        "description": ps.get("descriptionModule", {}).get("briefSummary"),

        # Enrollment
        "actual_enrollment": design.get("enrollmentInfo", {}).get("count")
            if design.get("enrollmentInfo", {}).get("type") == "ACTUAL" else None,

        # FDA Regulatory Oversight
        "is_fda_regulated_drug": oversight.get("isFdaRegulatedDrug"),
        "is_fda_regulated_device": oversight.get("isFdaRegulatedDevice"),

        # Oversight Quality
        "oversight_has_dmc": oversight.get("oversightHasDmc"),
        "oversight_authorities": oversight.get("oversightAuthorities", []),

        # Expanded Access
        "has_expanded_access": status.get("expandedAccessInfo", {}).get("hasExpandedAccess"),
    }

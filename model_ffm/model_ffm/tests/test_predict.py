import pytest
from model_ffm.predict import predict


class TestPredict:
    def test_full_func(self, sample_output, change_working_dir):
        columns = [
            "Standalone AD&D Coverage_ELECTION 'EE', 'ES', 'EC', or 'EF'",
            'Hospital Indemnity Coverage_ELECTION "EE", "ES", "EC", or "EF"',
            'Hospital Indemnity ELECTION "EE", "ES", "EC", or "EF"',
            'Hospital Indemnity ELECTION "EE", "ES", "EC", or "EF"_1',
            'Hospital Indemnity Coverage_ELECTION "EE", "ES", "EC", or "EF"_1',
            "Hospital Indemnity Coverage_ELECTION 'EE', 'ES', 'EC', or 'EF'",
            'Hospital Indemnity Accident_ELECTION "EE", "ES", "EC", or "EF"',
            "Hospital Indemnity ELECTION",
            "Hospital Indemnity.Election",
            'Hospital Coverage_ELECTION "EE", "ES", "EC", or "EF"',
            'Hospital ELECTION "EE", "ES", "EC", or "EF"',
            'Hospital ELECTION "EE", "ES", "EC", or "EF"_1',
            'Hospital Coverage_ELECTION "EE", "ES", "EC", or "EF"_1',
            "Hospital Coverage_ELECTION 'EE', 'ES', 'EC', or 'EF'",
            'Hospital.ELECTION "EE", "ES", "EC", or "EF"',
            "Hospital ELECTION",
            "Hospital.Election",
            'Hosp Coverage_ELECTION "EE", "ES", "EC", or "EF"_1',
            "Hosp Coverage_ELECTION 'EE', 'ES', 'EC', or 'EF'",
            'Hos_Coverage_ELECTION "EE", "ES", "EC", or "EF"',
            'Hos. ELECTION "EE", "ES", "EC", or "EF"',
            'Hos. ELECTION "EE", "ES", "EC", or "EF"_1',
            'Hospital_Indemnity_ELECTION "Plan 1, Plan 2"',
            "Hospital Indemnity Dual Option Plan 1 or 2_nan",
            'Hospital_ELECTION "Plan 1, Plan 2"',
            'Hospital.ELECTION "Plan 1, Plan 2"',
            "Hospital.Plan 1 or Plan 2",
            "Hospital Product Codes",
            "Hospital Indemnity_Effective Date",
            "Hospital Indemnity_Termination Date",
            "Hospital Indemnity Insurance_Benefit Amount",
            "Hospital Indemnity Insurance_Effective Date",
            "Hospital_Effective Date",
            "Hospital Monthly Premium",
            "Hospital.Benefit Amount",
            "Hospital.Plan 1 or Plan 2",
            "Hospital Indemnity Dual Option Plan 1 or 2.nan",
            "Hospital ELECTION",
            'Hospital_Indemnity_ELECTION "Plan 1, Plan 2"',
            "Dependent CHILD #3 SSN",
            "Member_Information_Last_Name",
            "Child_information_(1_age",
            "Child#1 DOB",
            "Child 2 DOB",
            "Ch1.LastName",
            "ACC Effective Date",
            "Member_Information_Employee_Benefit_Class",
            "Employee_Address_1",
            "Member_Information_Last_Name",
            "blank_header_1",
            "blank_header_20",
        ]
        results = predict(
            model_name="model_ffm",
            inputs={"datasetId": "spr:dataset_id", "columns": columns},
        )
        temp_res = list(map(lambda x: list(x.values())[0], results[0]['predictedResult']))
        temp_out = list(map(lambda x: list(x.values())[0], sample_output[0]['predictedResult']))

        res = [tuple(map(lambda x: x[0], lst)) for lst in temp_res]
        out = [tuple(map(lambda x: x[0], lst)) for lst in temp_out]

        assert res == out  # Note: test dont exactly match because of the slight differences in model training.
        # manually use -vv flag to check if the outputs are inline with training data

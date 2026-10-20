# HMDA — Ground-Truth de Constraints (rascunho)

Fonte: `cfpb/hmda-rule-spec` (FFIEC/CFPB), edição 2014 do LAR. Cada *edit* traz descrição em linguagem natural (o "texto normativo") + lógica formal (o gold). **63 validity + 32 quality = 95 constraints.**

Notação DC = padrão de violação (tuplas que NÃO devem existir). Nomes de campos = identificadores internos da spec (mapear para colunas públicas do LAR: `loanType`→`loan_type`, etc.).

> **validity** = bloqueantes (gold de extração/classificação; não geram violação nos dados publicados).  
> **quality** = warnings (gold de *violação* — sobrevivem no Modified LAR público).

## Resumo por categoria

| Categoria | validity | quality | total |
|---|--:|--:|--:|
| Domain / enumeration (IN) | 15 | 1 | 16 |
| Null/NA check (conditional) | 8 | 7 | 15 |
| Conditional selection cross-field (~CFD) | 25 | 10 | 35 |
| Range (conditional numeric) | 1 | 12 | 13 |
| Range (numeric) | 2 | 0 | 2 |
| Column comparison / ordering | 1 | 0 | 1 |
| Pattern/format (regex/LIKE) | 4 | 0 | 4 |
| Referential / external lookup | 4 | 0 | 4 |
| Distinctness (multi-valued) | 3 | 0 | 3 |
| Arithmetic (EXPRESSIVITY LIMIT) | 0 | 2 | 2 |
| **TOTAL** | **63** | **32** | **95** |


## Domain / enumeration (IN)  (16)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V220 | validity | Loan type must = 1, 2, 3, or 4. | `¬( t.loanType ∉ ['1', '2', '3', '4'] )` | loanType |
| V400 | validity | Property type must = 1, 2, or 3. | `¬( t.propertyType ∉ ['1', '2', '3'] )` | propertyType |
| V225 | validity | Loan purpose must = 1, 2, or 3. | `¬( t.loanPurpose ∉ ['1', '2', '3'] )` | loanPurpose |
| V230 | validity | Occupancy must = 1, 2, or 3. | `¬( t.ownerOccupancy ∉ ['1', '2', '3'] )` | ownerOccupancy |
| V415 | validity | Preapproval must = 1, 2, or 3. | `¬( t.preapprovals ∉ ['1', '2', '3'] )` | preapprovals |
| V255 | validity | Action taken type must = 1, 2, 3, 4, 5, 6, 7, or 8. | `¬( t.actionTaken ∉ ['1', '2', '3', '4', '5', '6', '7', '8'] )` | actionTaken |
| V450 | validity | Applicant ethnicity must = 1, 2, 3, or 4. | `¬( t.applicantEthnicity ∉ ['1', '2', '3', '4'] )` | applicantEthnicity |
| V460 | validity | Co-Applicant ethnicity must = 1, 2, 3, 4, or 5. | `¬( t.coapplicantEthnicity ∉ ['1', '2', '3', '4', '5'] )` | coapplicantEthnicity |
| V310 | validity | Applicant race field 1 must = 1, 2, 3, 4, 5, 6, or 7. | `¬( t.applicantRace1 ∉ ['1', '2', '3', '4', '5', '6', '7'] )` | applicantRace1 |
| V315 | validity | Co-Applicant race field 1 must = 1, 2, 3, 4, 5, 6, 7, or 8. | `¬( t.coapplicantRace1 ∉ ['1', '2', '3', '4', '5', '6', '7', '8'] )` | coapplicantRace1 |
| V320 | validity | Applicant sex must = 1, 2, 3, or 4. | `¬( t.applicantSex ∉ ['1', '2', '3', '4'] )` | applicantSex |
| V325 | validity | Co-Applicant sex must = 1, 2, 3, 4, or 5. | `¬( t.coapplicantSex ∉ ['1', '2', '3', '4', '5'] )` | coapplicantSex |
| V340 | validity | Type of purchaser must = 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9. | `¬( t.purchaserType ∉ ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] )` | purchaserType |
| V525 | validity | HOEPA status must = 1 or 2. | `¬( t.hoepaStatus ∉ ['1', '2'] )` | hoepaStatus |
| V550 | validity | Lien status must = 1, 2, 3, or 4. | `¬( t.lienStatus ∉ ['1', '2', '3', '4'] )` | lienStatus |
| Q022 | quality | Date application received should = activity year or (activity year minus 1 or activity year minus 2). | `¬( ¬constraint )  [see logic]` | applicationReceivedDate |

## Null/NA check (conditional)  (15)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V215 | validity | If Action taken type = 6, then date application received must = NA. | `¬( t.actionTaken = 6 ∧ ¬(t.applicationReceivedDate = NA) )` | actionTaken, applicationReceivedDate |
| V262 | validity | If date application received = NA, then action taken type must = 6. | `¬( t.applicationReceivedDate = NA ∧ ¬(t.actionTaken = 6) )` | actionTaken, applicationReceivedDate |
| V335 | validity | If property type = 3, then income must = NA. | `¬( t.propertyType = 3 ∧ ¬(t.applicantIncome = NA) )` | applicantIncome, propertyType |
| V338 | validity | If applicant ethnicity, race, and sex = 4, 7, and 4, respectively, and co-applicant ethnicity, race, and sex = 5, 8, and 5, respectively, and action taken = 1-5 or 7-8 then applicant income must = NA. | `¬( (complex) ∧ ¬(t.applicantIncome = NA) )` | actionTaken, applicantEthnicity, applicantIncome, applicantRace1, applicantSex, coapplicantEthnicity, coapplicantRace1, coapplicantSex |
| V505 | validity | If action taken type = 2-8, then rate spread must = NA. | `¬( t.actionTaken between  ∧ ¬(t.rateSpread = NA) )` | actionTaken, rateSpread |
| V570 | validity | If lien status = 1, then rate spread must be ≥ 1.5% and ≤ 99.99% or NA. | `¬( t.lienStatus = 1 ∧ ¬((complex)) )` | lienStatus, rateSpread |
| V575 | validity | If lien status = 2, then rate spread must be ≥ 3.5% and ≤ 99.99% or NA. | `¬( t.lienStatus = 2 ∧ ¬((complex)) )` | lienStatus, rateSpread |
| V520 | validity | If lien status = 3, then rate spread must = NA. | `¬( t.lienStatus = 3 ∧ ¬(t.rateSpread = NA) )` | lienStatus, rateSpread |
| Q027 | quality | If action taken type = 1-5, 7 or 8, and property type = 1 or 2, then applicant income should not = NA. | `¬( (complex) ∧ ¬(t.applicantIncome ≠ NA) )` | actionTaken, applicantIncome, propertyType |
| Q067 | quality | If applicant ethnicity, race, and sex = 4, 7, and 4, respectively, and co-applicant ethnicity, race, and sex = 4, 7, and 4, respectively, then applicant income should = NA. | `¬( (complex) ∧ ¬(t.applicantIncome = NA) )` | applicantEthnicity, applicantIncome, applicantRace1, applicantSex, coapplicantEthnicity, coapplicantRace1, coapplicantSex |
| Q039 | quality | If HOEPA status = 1 and action taken type = 1, then rate spread should not = NA. | `¬( (complex) ∧ ¬(t.rateSpread ≠ NA) )` | actionTaken, hoepaStatus, rateSpread |
| Q040 | quality | If purchaser type = 1-4 and lien status = 1 or 2, then rate spread should be ≤ 10% or NA. | `¬( (complex) ∧ ¬((complex)) )` | lienStatus, purchaserType, rateSpread |
| Q061 | quality | If agency = 5, property type = 1, lien status = 1 and action taken type = 1, rate spread should be ≤ 5% or NA. | `¬( (complex) ∧ ¬((complex)) )` | actionTaken, agencyCode, lienStatus, propertyType, rateSpread |
| Q066 | quality | If rate spread does not = NA, then rate spread should be < 13%. | `¬( t.rateSpread ≠ NA ∧ ¬(t.rateSpread < 13.0) )` | rateSpread |
| Q049 | quality | If action taken type = 7 or 8, then MSA/MD, state, county, census tract should equal NA. | `¬( t.actionTaken ∈ ['7', '8'] ∧ ¬((complex)) )` | actionTaken, censusTract, fipsCounty, fipsState, metroArea |

## Conditional selection cross-field (~CFD)  (35)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V410 | validity | If lien status = 3, then loan purpose must = 2. | `¬( t.lienStatus = 3 ∧ ¬(t.loanPurpose = 2) )` | lienStatus, loanPurpose |
| V425 | validity | If loan purpose = 1 and action taken type = 6, then preapproval must = 3. | `¬( (complex) ∧ ¬(t.preapprovals = 3) )` | actionTaken, loanPurpose, preapprovals |
| V430 | validity | If loan purpose = 2 or 3, then preapproval must = 3. | `¬( t.loanPurpose ∈ ['2', '3'] ∧ ¬(t.preapprovals = 3) )` | loanPurpose, preapprovals |
| V435 | validity | If action taken type = 7 or 8, then preapproval must = 1. | `¬( t.actionTaken ∈ ['7', '8'] ∧ ¬(t.preapprovals = 1) )` | actionTaken, preapprovals |
| V440 | validity | If preapproval = 1, then action taken type must = 1-5, 7 or 8. | `¬( t.preapprovals = 1 ∧ ¬(t.actionTaken ∈ ['1', '2', '3', '4', '5', '7', '8']) )` | actionTaken, preapprovals |
| V445 | validity | If preapproval = 2, then action taken type must = 1-5. | `¬( t.preapprovals = 2 ∧ ¬(t.actionTaken ∈ ['1', '2', '3', '4', '5']) )` | actionTaken, preapprovals |
| V447 | validity | If preapproval = 3, then action taken type must = 1-6. | `¬( t.preapprovals = 3 ∧ ¬(t.actionTaken ∈ ['1', '2', '3', '4', '5', '6']) )` | actionTaken, preapprovals |
| V463 | validity | Co-Applicant ethnicity does not = 5; however, co-applicant race field 1 = 8 and/or co-applicant sex = 5. | `¬( (complex) ∧ ¬(t.coapplicantEthnicity = 5) )` | coapplicantEthnicity, coapplicantRace1, coapplicantSex |
| V455 | validity | If applicant ethnicity = 1, 2 or 3 then the first applicant race field must not = 7. | `¬( t.applicantEthnicity ∈ ['1', '2', '3'] ∧ ¬(t.applicantRace1 ≠ 7) )` | applicantEthnicity, applicantRace1 |
| V465 | validity | If co-applicant ethnicity = 1, 2 or 3 then the first co-applicant race field must not = 7 or 8. | `¬( t.coapplicantEthnicity ∈ ['1', '2', '3'] ∧ ¬(t.coapplicantRace1 not_in ['7', '8']) )` | coapplicantEthnicity, coapplicantRace1 |
| V470 | validity | If applicant race = 1-5 in applicant race field 1, then all other applicant race fields must = blank or 1-5. | `¬( t.applicantRace1 ∈ ['1', '2', '3', '4', '5'] ∧ ¬((complex)) )` | applicantRace1, applicantRace2, applicantRace3, applicantRace4, applicantRace5 |
| V475 | validity | If applicant race = 6 or 7 in applicant race field 1, then all other applicant race fields must = blank. | `¬( t.applicantRace1 ∈ ['6', '7'] ∧ ¬((complex)) )` | applicantRace1, applicantRace2, applicantRace3, applicantRace4, applicantRace5 |
| V317 | validity | Co-Applicant race field 1 does not = 8; however, co-applicant ethnicity and/or co-applicant sex = 5. | `¬( (complex) ∧ ¬(t.coapplicantRace1 = 8) )` | coapplicantEthnicity, coapplicantRace1, coapplicantSex |
| V485 | validity | If co-applicant race = 1-5 in co-applicant race field 1, then all other co-applicant race fields must = blank or 1-5. | `¬( t.coapplicantRace1 ∈ ['1', '2', '3', '4', '5'] ∧ ¬((complex)) )` | coapplicantRace1, coapplicantRace2, coapplicantRace3, coapplicantRace4, coapplicantRace5 |
| V490 | validity | If co-applicant race = 6, 7, or 8 in co-applicant race field 1, then all other co-applicant race fields must = blank. | `¬( t.coapplicantRace1 ∈ ['6', '7', '8'] ∧ ¬((complex)) )` | coapplicantRace1, coapplicantRace2, coapplicantRace3, coapplicantRace4, coapplicantRace5 |
| V326 | validity | Co-Applicant sex does not = 5; however, co-applicant ethnicity = 5 and/or co-applicant race field 1 = 8. | `¬( (complex) ∧ ¬(t.coapplicantSex = 5) )` | coapplicantEthnicity, coapplicantRace1, coapplicantSex |
| V347 | validity | If type of purchaser = 1, 2, 3, 4, 5, 6, 7, 8, or 9, then action taken type must be 1 or 6. | `¬( t.purchaserType ∈ ['1', '2', '3', '4', '5', '6', '7', '8', '9'] ∧ ¬(t.actionTaken ∈ ['1', '6']) )` | actionTaken, purchaserType |
| V375 | validity | If type of purchaser = 2, then loan type must = 2, 3, or 4. | `¬( t.purchaserType = 2 ∧ ¬(t.loanType ∈ ['2', '3', '4']) )` | loanType, purchaserType |
| V355 | validity | If (agency code = 2, 3, 5, 7, or 9) or (agency code = 1 and action taken not = 3 or 7), then reasons for denial must = 1, 2, 3, 4, 5, 6, 7, 8, 9, or blank. | `¬( (complex) ∧ ¬((complex)) )` | actionTaken, agencyCode, denialReason1, denialReason2, denialReason3 |
| V385 | validity | If agency code = 1 and action taken = 3 or 7, then at least one reason for denial must be provided and must = 1, 2, 3, 4, 5, 6, 7, 8, or 9. Other reasons for denial must = 1, 2, 3, 4, 5, 6, 7, 8, 9 or blank. | `¬( (complex) ∧ ¬((complex)) )` | actionTaken, agencyCode, denialReason1, denialReason2, denialReason3 |
| V540 | validity | If action taken type = 2-5, 7, or 8, HOEPA status must = 2. | `¬( t.actionTaken ∈ ['2', '3', '4', '5', '7', '8'] ∧ ¬(t.hoepaStatus = 2) )` | actionTaken, hoepaStatus |
| V545 | validity | If lien status = 3, then HOEPA status must = 2. | `¬( t.lienStatus = 3 ∧ ¬(t.hoepaStatus = 2) )` | hoepaStatus, lienStatus |
| V555 | validity | If loan purpose = 1 or 3, then lien status must = 1, 2, or 4. | `¬( t.loanPurpose ∈ ['1', '3'] ∧ ¬(t.lienStatus ∈ ['1', '2', '4']) )` | lienStatus, loanPurpose |
| V560 | validity | If action taken type = 1-5, 7 or 8, then lien status must = 1, 2, or 3. | `¬( t.actionTaken ∈ ['1', '2', '3', '4', '5', '7', '8'] ∧ ¬(t.lienStatus ∈ ['1', '2', '3']) )` | actionTaken, lienStatus |
| V565 | validity | If action taken type = 6, then lien status must = 4. | `¬( t.actionTaken = 6 ∧ ¬(t.lienStatus = 4) )` | actionTaken, lienStatus |
| Q035 | quality | If purchaser type = 1 or 3, then loan type should = 1. | `¬( t.purchaserType ∈ ['1', '3'] ∧ ¬(t.loanType = 1) )` | loanType, purchaserType |
| Q032 | quality | If action taken type = 1, then action taken date should not equal the date application received. | `¬( t.actionTaken = 1 ∧ ¬(t.actionDate not_equal_property applicationReceivedDate) )` | actionDate, actionTaken |
| Q026 | quality | If action taken type = 1-5, 7, or 8, then applicant ethnicity, race and/or sex should not = 4, 7, or 4, respectively. | `¬( t.actionTaken ∈ ['1', '2', '3', '4', '5', '7', '8'] ∧ ¬((complex)) )` | actionTaken, applicantEthnicity, applicantRace1, applicantSex |
| Q068 | quality | If action taken type = 1-5, 7 or 8, and applicant ethnicity, race and sex = 4,7,4, respectively, and co-applicant ethnicity, race and sex = 4,7,4, respectively, and there is no co-applicant, then co-applicant ethnicity, race, and sex should = 5,8,5, respectively. | `¬( (complex) ∧ ¬((complex)) )` | actionTaken, applicantEthnicity, applicantRace1, applicantSex, coapplicantEthnicity, coapplicantRace1, coapplicantSex |
| Q046 | quality | If loan purpose = 1, HOEPA Status should not = 1. | `¬( t.loanPurpose = 1 ∧ ¬(t.hoepaStatus ≠ 1) )` | hoepaStatus, loanPurpose |
| Q050 | quality | If agency = 5, HOEPA status should not = 1. | `¬( t.agencyCode = 5 ∧ ¬(t.hoepaStatus ≠ 1) )` | agencyCode, hoepaStatus |
| Q051 | quality | If applicant ethnicity, race and sex = 4, 7, and 4, respectively, HOEPA status should not = 1. | `¬( (complex) ∧ ¬(t.hoepaStatus ≠ 1) )` | applicantEthnicity, applicantRace1, applicantSex, hoepaStatus |
| Q052 | quality | If property type = 3, HOEPA status should not = 1. | `¬( t.propertyType = 3 ∧ ¬(t.hoepaStatus ≠ 1) )` | hoepaStatus, propertyType |
| Q064 | quality | If purchaser type = 1 or 3, HOEPA status should not = 1. | `¬( t.purchaserType ∈ ['1', '3'] ∧ ¬(t.hoepaStatus ≠ 1) )` | hoepaStatus, purchaserType |
| Q059 | quality | If loan type = 3 or 4, then property type should not = 3. | `¬( t.loanType ∈ ['3', '4'] ∧ ¬(t.propertyType ≠ 3) )` | loanType, propertyType |

## Range (conditional numeric)  (13)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V260 | validity | If reasons for denial are in the range 1-9, then action taken type must = 3 or 7. | `¬( (complex) ∧ ¬(t.actionTaken ∈ ['3', '7']) )` | actionTaken, denialReason1, denialReason2, denialReason3 |
| Q002 | quality | If property type = 1 and income ≤ $200 thousand, then loan amount should be < $2 million. | `¬( (complex) ∧ ¬(t.loanAmount < 2000) )` | applicantIncome, loanAmount, propertyType |
| Q003 | quality | If loan type = 2 and property type = 1 or 2, then loan amount should be ≤ $729 ($729 thousand). | `¬( (complex) ∧ ¬(t.loanAmount ≤ 729) )` | loanAmount, loanType, propertyType |
| Q004 | quality | If loan type = 3 and property type = 1 or 2, then loan amount should be ≤ $1,094 ($1,094 thousands). | `¬( (complex) ∧ ¬(t.loanAmount ≤ 1094) )` | loanAmount, loanType, propertyType |
| Q005 | quality | If type of purchaser = 1-4, and property type = 1 or 2, then loan amount should be ≤ $729 ($729 thousand). | `¬( (complex) ∧ ¬(t.loanAmount ≤ 729) )` | loanAmount, propertyType, purchaserType |
| Q013 | quality | If property type = 3, then loan amount should be in the range of $100 thousand and $10 million. | `¬( t.propertyType = 3 ∧ ¬(t.loanAmount between ) )` | loanAmount, propertyType |
| Q036 | quality | If property type = 2, then loan amount should be ≤ $150 ($150 thousand). | `¬( t.propertyType = 2 ∧ ¬(t.loanAmount ≤ 150) )` | loanAmount, propertyType |
| Q037 | quality | If lien status = 2, then loan amount should be ≤ $250 ($250 thousand). | `¬( t.lienStatus = 2 ∧ ¬(t.loanAmount ≤ 250) )` | lienStatus, loanAmount |
| Q038 | quality | If lien status = 3, then loan amount should be ≤ $100 ($100 thousand). | `¬( t.lienStatus = 3 ∧ ¬(t.loanAmount ≤ 100) )` | lienStatus, loanAmount |
| Q025 | quality | If loan purpose = 1 and property type = 1, then loan amount should be > $10 thousand. | `¬( (complex) ∧ ¬(t.loanAmount > 10) )` | loanAmount, loanPurpose, propertyType |
| Q014 | quality | If income is numeric, then income should be < $3 million. | `¬( t.applicantIncome is_integer  ∧ ¬(t.applicantIncome < 3000) )` | applicantIncome |
| Q044 | quality | If agency = 1, 2, 3, 7 or 9, action taken type = 1, lien status = 1, and rate spread > 6.5%, then HOEPA status should = 1. | `¬( (complex) ∧ ¬(t.hoepaStatus = 1) )` | actionTaken, agencyCode, hoepaStatus, lienStatus, rateSpread |
| Q045 | quality | If agency = 1, 2, 3, 7 or 9, action taken type = 1, lien status = 2, and rate spread > 8.5%, then HOEPA status should = 1. | `¬( (complex) ∧ ¬(t.hoepaStatus = 1) )` | actionTaken, agencyCode, hoepaStatus, lienStatus, rateSpread |

## Range (numeric)  (2)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V250 | validity | Loan amount must be numeric and > zero. | `¬( ¬constraint )  [see logic]` | loanAmount |
| V330 | validity | Income must be numeric and > 0, or equal NA. | `¬( ¬constraint )  [see logic]` | applicantIncome |

## Column comparison / ordering  (1)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V275 | validity | If date application received does not = NA, then action taken date must be ≥ date application received. | `¬( t.applicationReceivedDate ≠ NA ∧ ¬(t.actionDate greater_than_or_equal_property applicationReceivedDate) )` | actionDate, applicationReceivedDate |

## Pattern/format (regex/LIKE)  (4)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V210 | validity | Date application received must be in CCYYMMDD format; month and day must be in the range 01-12 and 01-31, respectively or equal NA. CCYY must be numeric and CC (century) must equal 20. | `¬( t.applicationReceivedDate ≠ NA ∧ ¬((complex)) )` | applicationReceivedDate |
| V265 | validity | Action taken date must be in CCYYMMDD format; month and day must be in the range 01-12 and 01-31, respectively. CCYY must be numeric. | `¬( ¬constraint )  [see logic]` | actionDate |
| V300 | validity | Census tract must = a valid census tract number for the MSA/MD, state, county combination or (NA if county is classified as small) or (where MSA/MD = NA the census tract must = a valid census tract for the state/county combination or NA). Valid census tract format must be NNNN.NN or NA, left justified. | `¬( ¬constraint )  [see logic]` | censusTract |
| V500 | validity | Rate spread must be in the NN.NN format, must not = blank, and if NA, left justified. | `¬( ¬constraint )  [see logic]` | rateSpread |

## Referential / external lookup  (4)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V280 | validity | MSA/MD must = a valid Metropolitan Statistical Area or Metropolitan Division (if appropriate) code for period being processed or NA. | `¬( ¬constraint )  [see logic]` | metroArea |
| V285 | validity | State must = a valid FIPS code or (NA where MSA/MD = NA). | `¬( ¬constraint )  [see logic]` | fipsState, metroArea |
| V290 | validity | If MSA/MD does not = NA, then MSA/MD, state, and county codes must = a valid combination. | `¬( t.metroArea ≠ NA ∧ ¬(t.metroArea call ) )` | metroArea |
| V295 | validity | State and county must = a valid combination or (county = NA where MSA/MD = NA). | `¬( ¬constraint )  [see logic]` | fipsCounty, fipsState, metroArea |

## Distinctness (multi-valued)  (3)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| V480 | validity | Applicant race must not be the same (i.e. 1,1; 1,1,2; 1,2,3,4,1), when more than one applicant race is designated. | `¬( ¬constraint )  [see logic]` | applicantRace2, applicantRace3, applicantRace4, applicantRace5 |
| V495 | validity | Co-Applicant race must not be the same (i.e. 1,1; 1,1,2; 1,2,3,4,1), when more than one co-applicant race is designated. | `¬( ¬constraint )  [see logic]` | coapplicantRace2, coapplicantRace3, coapplicantRace4, coapplicantRace5 |
| V360 | validity | Responses for reasons for denial must not be the same (i.e. 1,1; 1,1,2; 1,2,2;...........). | `¬( ¬constraint )  [see logic]` | denialReason2, denialReason3 |

## Arithmetic (EXPRESSIVITY LIMIT)  (2)

| ID | Kind | Descrição (texto normativo) | DC (padrão de violação) | Campos |
|---|---|---|---|---|
| Q001 | quality | If loan amount and income are numeric and > 0 and the loan amount is ≥ $1,000 ($1 million), then loan amount should be < 5 times the income. | `¬( ¬constraint )  [see logic]` | loanAmount |
| Q024 | quality | If action taken type = 1, and income and loan amount are numeric, and loan amount is ≥ 5 times income, then income should be > $ 9 thousand. | `¬( (complex) ∧ ¬(t.applicantIncome > 9) )` | actionTaken, applicantIncome, loanAmount |
---

## Checabilidade de violação no Modified LAR público

Verificado contra a lista de campos redacted/modificados do Modified LAR (FFIEC):

- **Campos públicos e usáveis:** loan_type, loan_purpose, action_taken, income, lien_status, rate_spread, hoepa_status, occupancy, derived_msa, state, county, census_tract, ethnicity, race, sex, preapproval, purchaser_type, loan_amount (arredondado p/ múltiplos de $10k).
- **Campos redacted:** datas (application_received_date, action_date) e denial reasons → **11 edits** ficam fora da *violation detection* sobre dados públicos: V210, V215, V260, V262, V265, V275, V355, V360, V385, Q022, Q032.

**=> 84/95 edits checáveis em violação sobre o Modified LAR público.** Os 11 restantes continuam válidos como gold de *extração/classificação* (texto→constraint não precisa dos dados). Alternativa: usar dados HMDA não-modificados (pré-publicação) ou histórico para recuperar datas/denial.

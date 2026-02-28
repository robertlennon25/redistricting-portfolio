2020 PL 94-171 Data for New York based on the Decennial Census at the Block level on 2020 Census Redistricting Data (P.L. 94-171) Shapefiles

Please note that we have NOT validated against the official data used by your state’s redistricting body or bodies. Some states reallocate incarcerated persons and/or exclude non-permanent residents from the PL 94-171 data file for redistricting. Other states may make additional modifications. For more information about state modifications, visit our PL 94-171 Modifications article included in https://redistrictingdatahub.org/data/about-our-data/pl-94171-dataset/states-and-modification-wording/

##Redistricting Data Hub (RDH) Retrieval Date
08/12/21

##Sources
2020 PL 94-171 Legacy Format Data was retrieved from the US Census FTP site https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/
2020 Census Redistricting Data (P.L. 94-171) Shapefiles were retrieved on 05/24/21 from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

##Fields
For a full list of fields and descriptions, review the 2020 Census State Redistricting Data (Public Law 94-171) Summary File Technical Documentation at https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/complete-tech-docs/summary-file/2020Census_PL94_171Redistricting_StatesTechDoc_English.pdf

##Processing
The legacy format 2020 PL 94-171 Data was downloaded from the Census. 
The legacy format data is provided in one zip file per state. Each zip file contains four files: 3 “segments” containing the data for 1 or more standard redistricting tables, and 1 “geographic header” file. 
This first segment contains the data for Tables P1 (Race) and P2 (Hispanic or Latino, and Not Hispanic or Latino by Race). The second segment contains data for Tables P3 (Race for the Population 18 Years and Over), P4 (Hispanic or Latino, and Not Hispanic or Latino by Race for the Population 18 Years and Over), and H1 (Occupancy Status). The third segment contains Table P5 (Group Quarters Population by Major Group Quarters Type), which was not part of the 2010 PL 94-171 data release.
The files were imported into Python as pipe-delimited data frames and the columns renamed. The segments were queried by table (P1, P2, P3, P4, P5, H1, but P5 and H1 are joined together due to being small tables) and then each table was joined to the geo file, using the logical record number, or LOGRECNO.
For size reasons, we kept only a subset of the the geo file including only LOGRECNO, GEOID, and the SUMLEV (Summary Level). 
The block level data were queried from the dataframes and were merged with the 2020 Census Redistricting Data (P.L. 94-171) block shapefiles based on Census GEOIDs. 
The data were then extracted to shapefiles. There are five shapefiles (and their supporting files) in this folder one for each table (P1, P2, P3, P4) except H1 and P5 which are combined into one file. 
The RDH verified results internally and externally with partner organizations. 
For additional processing information, review our GitHub https://github.com/nonpartisan-redistricting-datahub

##Additional Notes
For more information about this data set, visit our PL 94-171 article included in https://redistrictingdatahub.org/data/about-our-data/. 

For additional questions, email info@redistrictingdatahub.org.
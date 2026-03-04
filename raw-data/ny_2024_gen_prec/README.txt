New York 2024 General Election Precinct-Level Results and Boundaries

## RDH Date Retrieval
02/09/26

## Sources
Precinct boundaries and election results from Ben Rosenblatt (https://www.benjrosenblatt.com/). 

Precinct boundaries were checked against a voter file from L2 dated October 26th, 2024. 

## Notes on Field Names:
Columns reporting votes generally follow the pattern:
One example is:
GPREDCLI
The first character is G for a general election, P for a primary, S for a special, and R for a runoff.
Characters 2 and 3 are the year of the election.
Characters 4-6 represent the office type (see list below).
Character 7 represents the party of the candidate.
Characters 8-10 are the first three letters of the candidate's last name.

Office Codes Used:
CON - US Congress
PRE - President
PRO - Statewide Proposition
SL - State Assembly
SU - State Senate
USS - US Senate

Party Codes Used:
D - Democratic
L - LaRouche
O - Other / Write-in
R - Republican

## Fields:
Field Name Description               
UNIQUE_ID  Precinct Unique Identifier
COUNTYFP   County FIP Code           
GEOID24    Precinct GEOID            
CountyName County Name               
MuniName   Municipality Name         
MuniID     Municipality ID           
DivType    Division Type             
DivID      Division ID               
EDID       Election District ID      
EDName     Election District Name    
Locality   Locality                  
LocalGEOID Locality GEOID            
Congress   Congressional District    
Senate     State Senate District     
Assembly   State Assembly District   
G24COND    U.S. Congress, Democratic Candidate Combined Votes                          
G24CONO    U.S. Congress, Other Votes                          
G24CONR    U.S. Congress, Republican Candidate Combined Votes                          
G24PREDHAR Presidential, Kamala Harris Votes (DEM + WFP)                    
G24PREOWRI Presidential, Other / Write-in Candidate Votes                          
G24PRERTRU Presidential, Donald Trump Votes (REP + CON)                        
G24PRO1NO  Statewide Proposition 1, No Votes                          
G24PRO1YES Statewide Proposition 1, Yes Votes                          
G24SLD     State Assembly, Democratic Candidate Combined Votes                          
G24SLO     State Assembly, Other Votes                          
G24SLR     State Assembly, Republican Candidate Combined Votes                          
G24SUD     State Senate, Democratic Candidate Combined Votes                          
G24SUO     State Senate, Other Votes                        
G24SUR     State Senate, Republican Candidate Combined Votes                          
G24USSDGIL U.S. Senate, Kirsten Gillibrand Votes (DEM + WFP)                          
G24USSRSAP U.S. Senate, Mike Sapraicone Votes (REP + CON)                         
G24USSLSAR U.S. Senate, Diane Sare Votes (LaRouche)                         
G24USSOWRI U.S. Senate, Other / Write-in Candidate Votes                          

## Notes:
In cases where candidates were nominated by two different parties (typically Working Families Party and Democratic Party or Conservative Party and Republican Party) votes are combined and referred to as "Democratic" or "Republican" votes respectively.

There are a handful of differences between the precinct-level results and the county-level results available from the New York Secretary of State. The most significant of this differences relates to Nassau County State Senate votes in District 9, for which a significant number (~30k) of votes do not appear to be available at the precinct-level. We are in the process of digging into this discrepancy and figuring out whether updates to this file can be made. 

Around 30 ballots in Jefferson, Schulyer and Yates counties are not tied to specific precincts. These votes are not included in the joined file.

Please direct questions related to processing this dataset to info@redistrictingdatahub.org.

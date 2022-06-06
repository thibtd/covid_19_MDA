
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/moxious/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/$file' AS line

WITH coalesce(line.`Province/State`, 'General') as province,
     coalesce(line.`Country/Region`, 'Missing') as region,
     datetime(line.`Last Update`) as lastUpdate,
     date(datetime({ epochMillis: apoc.date.parse(replace("$file", ".csv", ""), 'ms', 'MM-dd-yyyy') })) as reportDate,
     toFloat(line.Latitude) as latitude,
     toFloat(line.Longitude) as longitude,
     line

MERGE (p:Province { name: province })
MERGE (r:Region { name: region })
CREATE (s:Report {
    label: region + ' ' + line.Confirmed + ' CONFIRMED ' + line.Deaths + ' Deaths ' + line.Recovered + ' Recovered',

    /* Critical distinction: reportDate is when the stat was reported, lastUpdate was when the province provided
     * the data.  So for example, on March 2 2020, if the last update available from San Mateo, CA was on February 28th,
     * then lastUpdate=February 28, and reportDate=March 2
     */
    reportDate: reportDate,
    lastUpdate: lastUpdate,

    /* How far the province reporting is lagging.  3 means that the last stat we got was 3 days ago */
    ageInDays: duration.between(lastUpdate, reportDate).days,
    
    confirmed: toInteger(coalesce(line.Confirmed, 0)),
    deaths: toInteger(coalesce(line.Deaths, 0)),
    recovered: toInteger(coalesce(line.Recovered, 0)),
    latitude: latitude,
    longitude: longitude,
    location: point({ latitude: latitude, longitude: longitude })
})

MERGE (p)-[:IN]->(r)
CREATE (p)-[:REPORTED]->(s)
RETURN count(s) as ReportsLoaded;
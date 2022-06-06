
MATCH (r:Region), (c:Country)
WHERE r.name = c.name
MERGE (r)-[rel:IS]->(c)
RETURN count(rel) as RegionCountryLinks;

MATCH (p:Province), (c:City)
WHERE p.name = c.name
MERGE (p)-[rel:IS]->(c)
RETURN count(rel) as ProvinceCityLinks;

/* Link reoprts that have geo information to nearest cities */
MATCH (r:Report), (c:City)
WHERE 
    r.location is not null and 
    c.location is not null and
    distance(c.location, r.location) < 5000
MERGE (r)-[rel:NEARBY]->(c)
RETURN count(rel) as ReportNearbyCityLinks;
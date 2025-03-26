# Implementation Plan: Using Supabase Anime Table Instead of AniList API

## Background & Current Situation

Currently, our recommendation system fetches anime details directly from the AniList API when generating recommendations. This approach has revealed several limitations:

1. We're hitting AniList API rate limits frequently (429 errors)
2. Our retry mechanisms and request queue are adding complexity
3. Recommendation generation is slowed by network requests
4. User experience suffers when rate limits are hit

While we have a Supabase `anime` table with essential information (genres, tags, ratings, etc.), we're currently only using it to fetch anime IDs in our `fetchAnimeFromSupabase` function, not the full anime details needed for the recommendation process.

## Hybrid Approach Implementation

To balance database efficiency with comprehensive data coverage, we'll implement a hybrid approach that:

1. **Prioritizes Supabase**: Fetch all anime details from our database first
2. **Falls back to AniList API**: Only for anime not found in our database
3. **Maintains cache**: Cache API responses to minimize redundant external calls

## Implementation Steps

### Phase 1: Enhance the Supabase Data Access Layer

1. **Create Enhanced Anime Data Service:**
   - Create a new `animeDataService.ts` in the `src/services` directory
   - Implement methods to fetch complete anime details from Supabase
   - Add fallback mechanism to query AniList API for missing anime

2. **Update the `fetchAnimeFromSupabase` Function:**
   - Modify to fetch complete anime data (genres, tags, etc.), not just IDs
   - Add proper typing based on the Supabase schema
   - Implement pagination to handle large datasets efficiently

### Phase 2: Modify Recommendation Service

1. **Update `getAnimeDetails` Function:**
   - Modify to check Supabase first
   - Fall back to AniList API only when necessary
   - Cache results in both directions (API → DB and DB → memory)

2. **Refactor Batch Processing:**
   - Replace `getAnimeDetailsInBatches` with a database-first version
   - Optimize database queries to fetch multiple anime at once
   - Maintain controlled API requests for fallbacks

3. **Update Preference Extraction Logic:**
   - Modify how we extract genres and tags to work with Supabase data format
   - Ensure compatibility with both data sources (API and DB)

### Phase 3: Data Synchronization Strategy

1. **Implement Background Data Sync:**
   - Create a scheduled job to periodically fetch new/updated anime from AniList
   - Add to our Supabase database in batches during off-peak hours
   - Prioritize popular/trending anime for synchronization

2. **Data Refresh Strategy:**
   - Implement logic to update stale anime data (older than X days)
   - Refresh data for anime that users interact with frequently
   - Build an admin interface to trigger manual syncs if needed

### Phase 4: Error Handling and Fallbacks

1. **Graceful Degradation:**
   - Handle cases where anime isn't in our database and API is unavailable
   - Provide partial recommendations based on available data
   - Communicate limitations to users when appropriate

2. **Monitoring and Alerting:**
   - Track database coverage metrics (% of requested anime found in DB)
   - Monitor fallback API request frequency
   - Alert when fallback usage exceeds thresholds

## Technical Considerations

### Database Schema Compatibility
- Ensure our Supabase schema matches what's needed for recommendations
- Map Supabase fields to what the recommendation algorithm expects
- Handle any differences in data formats between AniList and our DB

### Performance Optimization
- Use efficient querying patterns (bulk lookups over individual requests)
- Consider caching frequent requests in memory
- Index critical fields in the Supabase table

### Rate Limit Management
- Maintain the request queue for AniList API fallbacks
- Use exponential backoff for retries
- Implement circuit breaker pattern to avoid repeated failures

## Implementation Timeline

1. **Week 1: Data Access Layer**
   - Build enhanced anime data service
   - Implement caching strategy
   - Create unit tests for the service

2. **Week 2: Recommendation Service Updates**
   - Refactor recommendation service to use the new data service
   - Update preference extraction logic
   - Fix any compatibility issues between data sources

3. **Week 3: Data Synchronization**
   - Implement background sync job
   - Add data refresh strategy
   - Set up monitoring for database coverage

4. **Week 4: Testing and Optimization**
   - Comprehensive testing
   - Performance optimization
   - Rollout strategy

## Future Considerations

1. **Expanding Our Database:**
   - Increase coverage of anime in our database
   - Add more detailed information (staff, studios, related media)
   - Consider implementing a more comprehensive sync with AniList

2. **Cache Warming:**
   - Proactively cache popular anime
   - Pre-compute common recommendation patterns
   - Optimize cache invalidation strategies

3. **User Analytics:**
   - Track which anime are frequently requested but missing from our DB
   - Prioritize adding those to our database
   - Use recommendation patterns to inform DB growth strategy

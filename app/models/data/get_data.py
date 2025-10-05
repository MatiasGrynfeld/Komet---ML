import aiohttp
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import shutil

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
MAX_EVENTS = 20000
TARGET_EVENTS = 15000
SAMPLES = float('inf')

output_folder = Path('./raw_data/')
shutil.rmtree(output_folder, ignore_errors=True)
output_folder.mkdir(parents=True, exist_ok=True)

base_query_params = {
    "format": "geojson",
    "orderby": "time",
    "eventtype": "earthquake",
    "minmagnitude": "4.5",
}

async def get_event_count(session: aiohttp.ClientSession, start_date: str, end_date: str) -> int:
    count_params = base_query_params.copy()
    count_params.update({
        "starttime": start_date,
        "endtime": end_date
    })
    
    async with session.get(COUNT_URL, params=count_params) as response:
        response.raise_for_status()
        text = await response.text()
        try:
            count = int(text.strip())
            print(f"Count for {start_date} to {end_date}: {count} events")
            return count
        except ValueError:
            try:
                import json
                error_data = json.loads(text)
                if "count" in error_data:
                    count = error_data["count"]
                    print(f"Count for {start_date} to {end_date}: {count} events (exceeds API limit)")
                    return count
                else:
                    raise ValueError("Count not found in error response")
            except json.JSONDecodeError:
                print(f"Error parsing count response: {text}")
                raise ValueError("Invalid response from count API")

async def download_data(session: aiohttp.ClientSession, start_date: str, end_date: str, expected_count: int) -> int:
    query_params = base_query_params.copy()
    query_params.update({
        "starttime": start_date,
        "endtime": end_date
    })
    
    async with session.get(BASE_URL, params=query_params) as response:
        response.raise_for_status()
        text = await response.text()

    file_path = output_folder / f"earthquake_data_{start_date}_{end_date}.json"

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    print(f"✓ Data downloaded: {start_date} to {end_date} ({expected_count} events)")
    return expected_count

async def find_optimal_end_date(session: aiohttp.ClientSession, start_date: datetime, max_end_date: datetime) -> datetime:
    left = start_date
    right = max_end_date
    best_end = start_date
    
    while left <= right:
        days_diff = (right - left).days
        mid = left + timedelta(days=days_diff // 2)
        
        if mid <= start_date:
            break
            
        start_str = start_date.strftime("%Y-%m-%d")
        mid_str = mid.strftime("%Y-%m-%d")
        
        count = await get_event_count(session, start_str, mid_str)
        
        if count <= MAX_EVENTS:
            best_end = mid
            if count >= TARGET_EVENTS * 0.8:
                break
            left = mid + timedelta(days=1)
        else:
            right = mid - timedelta(days=1)
        
        await asyncio.sleep(0.5)
    
    return best_end

async def download_data_in_chunks(start_year: int, end_year: int):
    current_date = datetime(start_year, 1, 1)
    final_date = datetime(end_year, 12, 31)
    
    chunk_number = 1
    total_samples_collected = 0
    
    # Create a single session for all requests
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while current_date < final_date and total_samples_collected < SAMPLES:
            print(f"\n--- Processing chunk {chunk_number} starting from {current_date.strftime('%Y-%m-%d')} ---")
            print(f"Progress: {total_samples_collected:,}/{SAMPLES:,} samples collected ({(total_samples_collected/SAMPLES)*100:.1f}%)")

            optimal_end = await find_optimal_end_date(session, current_date, final_date)
            
            start_str = current_date.strftime("%Y-%m-%d")
            end_str = optimal_end.strftime("%Y-%m-%d")
            
            chunk_count = await get_event_count(session, start_str, end_str)
            
            actual_count = await download_data(session, start_str, end_str, chunk_count)
            total_samples_collected += actual_count
            
            # Move to next chunk
            current_date = optimal_end + timedelta(days=1)
            chunk_number += 1
            
            if total_samples_collected >= SAMPLES:
                break
            
            await asyncio.sleep(1)
    
    if total_samples_collected >= SAMPLES:
        print(f"\n✅ COMPLETION REASON: Sample target reached")
    else:
        print(f"\n✅ COMPLETION REASON: Timespan finished ({start_year}-{end_year})")

async def main():
    await download_data_in_chunks(1900, 2025)

if __name__ == "__main__":
    asyncio.run(main())
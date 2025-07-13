#!/usr/bin/env python3
"""
Migration script to convert existing coordinate data to GeoJSON format
for MongoDB 2dsphere indexing compatibility.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import DB
import json

def migrate_coordinates():
    """Migrate existing coordinate data to GeoJSON format."""
    print("=== MIGRATING COORDINATES TO GEOJSON FORMAT ===")
    
    # Initialize database
    db = DB()
    
    # Get all users with old coordinate format
    users_to_migrate = []
    for user in db.users_collection.find():
        location = user.get('location', {})
        coordinates = location.get('coordinates', {})
        
        # Check if coordinates are in old format {lat: x, lng: y}
        if isinstance(coordinates, dict) and 'lat' in coordinates and 'lng' in coordinates:
            if coordinates['lat'] is not None and coordinates['lng'] is not None:
                users_to_migrate.append({
                    'user_id': user['user_id'],
                    'name': user['name'],
                    'old_coordinates': coordinates,
                    'location': location
                })
    
    print(f"Found {len(users_to_migrate)} users with old coordinate format")
    
    # Migrate each user
    migrated_count = 0
    for user in users_to_migrate:
        try:
            old_coords = user['old_coordinates']
            lat = float(old_coords['lat'])
            lng = float(old_coords['lng'])
            
            # Update to GeoJSON format
            update_data = {
                'location.coordinates': {
                    'type': 'Point',
                    'coordinates': [lng, lat]  # GeoJSON format: [longitude, latitude]
                },
                # Keep separate fields for backward compatibility
                'location.latitude': lat,
                'location.longitude': lng
            }
            
            result = db.users_collection.update_one(
                {'user_id': user['user_id']},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                migrated_count += 1
                print(f"✅ Migrated {user['name']} ({user['user_id'][:8]}...): [{lng}, {lat}]")
            else:
                print(f"⚠️  No changes for {user['name']} ({user['user_id'][:8]}...)")
                
        except Exception as e:
            print(f"❌ Error migrating {user['name']} ({user['user_id'][:8]}...): {str(e)}")
    
    print(f"\nMigration completed: {migrated_count}/{len(users_to_migrate)} users migrated")
    
    # Now try to create the geospatial indexes
    print("\n=== CREATING GEOSPATIAL INDEXES ===")
    try:
        # Drop existing indexes if they exist (in case of partial creation)
        try:
            db.users_collection.drop_index("location.coordinates_2dsphere")
        except:
            pass
        
        # Create 2dsphere index for location coordinates
        db.users_collection.create_index([("location.coordinates", "2dsphere")])
        print("✅ Created 2dsphere index for location.coordinates")
        
        # Create compound index for location + user_id
        db.users_collection.create_index([
            ("location.coordinates", "2dsphere"),
            ("user_id", 1)
        ])
        print("✅ Created compound geospatial index")
        
        # Create index for user embeddings
        db.embeddings_collection.create_index([("user_id", 1)])
        db.embeddings_collection.create_index([("created_at", -1)])
        print("✅ Created embedding collection indexes")
        
        print("\n🎉 All indexes created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating indexes: {str(e)}")
        return False
    
    return True

def test_geospatial_query():
    """Test the new geospatial query functionality."""
    print("\n=== TESTING GEOSPATIAL QUERIES ===")
    
    db = DB()
    
    # Test finding nearby users for Prasad
    try:
        nearby_users = db.find_nearby_users('40e28a2e-4248-4e4a-968d-a25ef42e8c0f', max_distance_km=100)
        print(f"✅ Found {len(nearby_users)} nearby users using optimized geospatial query")
        
        for user in nearby_users[:3]:  # Show first 3 results
            print(f"  - {user['name']}: {user['distance_km']}km away in {user.get('city', 'Unknown city')}")
            
    except Exception as e:
        print(f"❌ Error in geospatial query: {str(e)}")
        
        # Try fallback method
        try:
            print("Trying fallback method...")
            nearby_users = db._find_nearby_users_fallback('40e28a2e-4248-4e4a-968d-a25ef42e8c0f', max_distance_km=100)
            print(f"✅ Fallback method found {len(nearby_users)} nearby users")
        except Exception as e2:
            print(f"❌ Fallback method also failed: {str(e2)}")

if __name__ == "__main__":
    print("Starting coordinate migration...")
    success = migrate_coordinates()
    
    if success:
        test_geospatial_query()
        print("\n✅ Migration completed successfully!")
        print("You can now use optimized geospatial queries with MongoDB 2dsphere indexes.")
    else:
        print("\n❌ Migration failed. Please check the errors above.") 
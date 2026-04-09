// Runs only on first Mongo container init (empty volume).
const appDbName = process.env.MONGO_APP_DB || "aeganmediamontage";
const appDbUser = process.env.MONGO_APP_USER || "aegan_app_user";
const appDbPassword = process.env.MONGO_APP_PASSWORD || "change-me";

db = db.getSiblingDB(appDbName);

db.createUser({
  user: appDbUser,
  pwd: appDbPassword,
  roles: [{ role: "readWrite", db: appDbName }],
});

db.createCollection("users");
db.createCollection("video_jobs");
db.createCollection("password_reset_requests");

print(`Initialized database: ${appDbName}`);


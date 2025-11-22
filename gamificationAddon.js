// pointsRewards.js
class PointsRewards {
    constructor(user) {
        this.user = user; // user object { id, points, rewards: [] }
    }

    // Reward points for following nudge (good behavior)
    rewardNudgeAction(actionType) {
        const pointsMap = {
            'followed_nudge': 10,
            'avoided_impulsive_trade': 20,
            'journaled_trade': 5
        };
        const rewardPoints = pointsMap[actionType] || 0;
        this.user.points += rewardPoints;
        this.user.rewards.push({
            action: actionType,
            points: rewardPoints,
            date: new Date()
        });
        return rewardPoints;
    }

    // Redeem points for a reward
    redeem(pointsRequired) {
        if (this.user.points >= pointsRequired) {
            this.user.points -= pointsRequired;
            return true; // success
        }
        return false; // insufficient points
    }

    // Get user points
    getPoints() {
        return this.user.points;
    }

    // Get rewards history
    getRewards() {
        return this.user.rewards;
    }
}

// Example usage:
const user = { id: 1, points: 0, rewards: [] };
const rewardsModule = new PointsRewards(user);

// User follows a nudge
rewardsModule.rewardNudgeAction('followed_nudge');  // +10 points
rewardsModule.rewardNudgeAction('avoided_impulsive_trade');  // +20 points

console.log(rewardsModule.getPoints());  // Should log 30
console.log(rewardsModule.getRewards()); // List of reward actions

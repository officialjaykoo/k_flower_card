export const ruleSets = {
  A: {
    name: "통합 단일룰",
    description: "프로젝트 공통 단일 맞고 룰(7점부터 Go/Stop)",
    goMinScore: 7,
    ttakBonus: 1,
    ppukBonus: 1,
    jjobBonus: 1,
    bakMultipliers: { gwang: 2, pi: 2, mongBak: 2 },
    goStopMultiplier: 2,
    useEarlyStop: true
  }
};
